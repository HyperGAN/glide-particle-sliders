#!/usr/bin/env python3
"""Build blind fresh-v3 preference scenarios and unrelated binary controls.

All scenarios are original synthetic examples authored without consulting model
outputs or selecting examples by predictions. Preference labels implement an
instructor-defined concision rubric, not a real user's learned preference.

The 40 fresh preference groups contain 32 equally correct short/long pairs and
eight controls whose shorter reply is wrong or critically incomplete. Each has
two option orders. Twelve unrelated semantic cases also have both option orders.
Their positive and neutral requests are identical, preferred_slot is factual
ground truth, control=True, and concise_slot is a nominal description-length
marker for schema compatibility (canonical first breaks equal-length ties).
Unrelated rows are evaluation-only: their teacher is exactly the native neutral
response T=N, and they must never enter teacher-relative-SSE ranking or training.

The deterministic builder refuses to overwrite a different existing file.
Freeze and hash these bytes before any model evaluation; never revise them in
response to performance. Old sources, datasets, and artifacts remain untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SCHEMA = "glide-concise-pairs-v1"
NEUTRAL_INSTRUCTIONS = (
    "Which reply better answers the request in the context? Prefer accuracy, "
    "completeness, and helpfulness."
)
POSITIVE_INSTRUCTIONS = (
    "Which reply better answers the request in the context? Prefer a concise "
    "reply when it is accurate, complete, and helpful; do not sacrifice essential "
    "information or correctness for brevity."
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "concise-fresh-v3.json"


def pair(group, domain, state, concise, verbose, *, control=False):
    return dict(group=group, domain=domain, state=state, concise=concise,
                verbose=verbose, control=control)


SOURCE_PAIRS = [
    pair(
        "fresh3-field-bird-markings", "natural_history_records",
        "A fictional island fieldbook describes an alderlet with a white forehead patch, a charcoal back, and an unmarked amber tail. Write a description containing those three visible features, without adding size or behavior.",
        "The alderlet has a white forehead patch, a charcoal back, and an unmarked amber tail. These are the three visible features recorded in the fieldbook.",
        "The fieldbook records a white patch on the alderlet's forehead. Its back is charcoal, while its tail is amber with no markings. A description based on this entry should keep those three features together, without introducing a size estimate or any account of the bird's behavior.",
    ),
    pair(
        "fresh3-field-habitat-evidence", "natural_history_records",
        "During an imaginary reed survey, silver reedbells were observed beside the stream but none were recorded in the meadow. The notes do not establish where the species can survive. Summarize the observation without turning it into a claim about survival.",
        "Silver reedbells were recorded beside the stream, with none recorded in the meadow. This observation does not establish whether the species can survive in either habitat.",
        "The survey notes report silver reedbells beside the stream and no recorded reedbells in the meadow. That describes what the observers found in those two places. It does not establish the species' survival limits, so the summary should not claim that meadow conditions make survival impossible.",
    ),
    pair(
        "fresh3-field-leaf-transition", "natural_history_records",
        "A fictional shrub entry records bronze leaves in early spring and green leaves by midsummer. No leaf-fall observation was made. Describe the recorded color transition and make clear that shedding was not documented.",
        "The shrub's recorded leaves change from bronze in early spring to green by midsummer. The entry does not document leaf shedding or when it might occur.",
        "The entry begins with bronze leaves observed in early spring and later records green leaves by midsummer. Those observations describe a change in leaf color across the recorded period. There is no observation of leaf fall, so the entry cannot supply a shedding time or confirm that shedding occurred.",
    ),
    pair(
        "fresh3-field-fossil-attachment", "natural_history_records",
        "An excavation note describes a ribbed fossil fragment beside a flat fragment. The pieces were not joined, and their relationship is unknown. Explain what was found and what remains uncertain without calling them parts of one organism.",
        "A ribbed fossil fragment lay beside a flat fragment, but the pieces were not joined. Their relationship remains unknown, including whether they came from one organism.",
        "The note records two nearby finds: one ribbed fossil fragment and one flat fragment. They were not physically joined when found. Although they lay beside one another, the note leaves their relationship unresolved, so it does not establish that the fragments belonged to the same organism.",
    ),
    pair(
        "fresh3-field-life-stage-description", "natural_history_records",
        "The imaginary duskfin has a spotted chest as a juvenile and a plain chest as an adult. Both stages have a dark crown. Explain how the chest distinguishes the stages and which feature they share.",
        "Juvenile duskfins have spotted chests, while adults have plain chests. Both stages share a dark crown, so chest markings distinguish them in this fieldbook.",
        "Look at the chest markings to distinguish the two stages described here. A juvenile duskfin has a spotted chest, whereas an adult's chest is plain. The dark crown is present at both stages, so that shared feature does not separate juveniles from adults in this entry.",
    ),
    pair(
        "fresh3-field-seed-pod-state", "natural_history_records",
        "A botanical notebook names two fictional seed pods. The spiral pod is closed and green; the lantern pod is split open and brown. Provide a comparison that preserves both the opening state and the color of each.",
        "The spiral pod is closed and green; the lantern pod is split open and brown. They differ in both their opening state and recorded color.",
        "The notebook describes the spiral pod as still closed, with a green color. It describes the lantern pod differently: that pod is split open and brown. A comparison should retain both distinctions, pairing the closed state with green for spiral and the open state with brown for lantern.",
    ),
    pair(
        "fresh3-field-color-reference", "natural_history_records",
        "A feather sample looked orange under the tent lamp but copper beside the observer's reference card in daylight. The catalog uses the daylight reference-card reading. State the catalog color and distinguish it from the lamp impression.",
        "Catalog the feather as copper, using the daylight reference-card reading. Orange was its appearance under the tent lamp, which is not the catalog's chosen reading.",
        "The catalog color should be copper because the catalog uses the reading made beside the reference card in daylight. The orange appearance came from viewing the feather under the tent lamp. Keep that lamp impression separate from the daylight reading rather than entering orange as the catalog color.",
    ),
    pair(
        "fresh3-field-unknown-sound-source", "natural_history_records",
        "An observer heard a two-note whistle behind dense reeds. No animal was seen, and the sound's source was not identified. Write a record preserving what was heard without naming an animal or claiming a visual sighting.",
        "A two-note whistle was heard behind dense reeds. No animal was seen, and the source remains unidentified, so the record does not name an animal.",
        "The record can describe the sound as a two-note whistle coming from behind dense reeds. It should also retain the limits of the observation: no animal was seen, and the source was not identified. Naming a particular animal or recording a visual sighting would go beyond these notes.",
    ),
    pair(
        "fresh3-field-camera-trigger", "natural_history_records",
        "A camera log contains one midnight image of a fictional ridge-marten carrying a pale object. The object cannot be identified, and the image does not reveal the animal's destination. Describe the image and its two unresolved details.",
        "The midnight image shows a ridge-marten carrying a pale object. The object is unidentified, and the animal's destination cannot be determined from this image.",
        "The camera image, recorded at midnight, shows a ridge-marten with a pale object. The available image does not identify what that object is. It also gives no determination of where the animal is heading, leaving both the object's identity and the animal's destination unresolved.",
    ),
    pair(
        "fresh3-field-track-interruption", "natural_history_records",
        "A tracking sketch shows a line of small paired prints ending at a smooth stone patch. There are no readable prints across that patch. The sketch does not show whether the animal stopped or continued. Explain the interruption without inferring its movement.",
        "The paired prints end at a smooth stone patch where no prints are readable. The sketch does not establish whether the animal stopped or continued across it.",
        "The visible line of small paired prints reaches a smooth stone patch and then becomes unreadable. That is an interruption in the recorded tracks, not a confirmed stop by the animal. The sketch leaves open whether the animal stopped there or continued without leaving readable prints across the patch.",
    ),
    pair(
        "fresh3-field-lichen-photo-limit", "natural_history_records",
        "A branch photograph from April shows a pale lichen patch near a fork. The planned autumn photograph was never taken. Explain what the record establishes and why it cannot show change between April and autumn.",
        "The April photograph establishes a pale lichen patch near the branch fork. With no autumn photograph, this record cannot show whether that patch changed between seasons.",
        "The available photograph documents a pale lichen patch beside the branch fork in April. There is no matching autumn image because the planned photograph was not taken. Without that later observation, this record cannot establish a change in the patch between April and autumn or describe its autumn condition.",
    ),
    pair(
        "fresh3-field-nest-occupancy", "natural_history_records",
        "A fictional hollowbird nest was empty when inspected, but the observer found fresh lining inside it. No bird was observed adding that lining. Summarize the empty nest and lining without claiming who placed it or when.",
        "The inspected hollowbird nest was empty and contained fresh lining. No bird was seen adding it, so its maker and placement time were not established.",
        "At inspection, the hollowbird nest contained no bird, but fresh lining was present inside it. The observer did not see a bird place that material. The record therefore supports an empty nest with fresh lining while leaving who added the lining, and when it was added, unresolved.",
    ),
    pair(
        "fresh3-field-scent-versus-analysis", "natural_history_records",
        "A flower note says the observer described the scent as 'sweet.' No chemical analysis was performed. Explain the status of that description without treating it as a measured chemical composition.",
        "'Sweet' records the observer's description of the flower's scent. It is not a measured chemical composition, because no chemical analysis was performed for this note.",
        "The word 'sweet' is the observer's way of describing how the flower smelled. The note contains no chemical analysis, so that descriptive word should not be presented as a measured statement about chemical composition. It records a sensory impression rather than an analysis of the flower's chemicals.",
    ),
    pair(
        "fresh3-field-sample-separation", "natural_history_records",
        "A collection record places loose bark flakes in envelope P and a fallen flower in envelope Q. Neither envelope contains a whole plant. Describe the two samples while preserving their envelope labels and the collection's scope.",
        "Envelope P contains loose bark flakes, and envelope Q contains a fallen flower. These are separate samples; neither envelope holds a whole plant in this collection.",
        "The collection has two labeled samples. Loose bark flakes are stored in envelope P, while envelope Q holds the fallen flower. Keep those contents attached to their respective labels. Both samples are limited parts recorded by the collection; neither envelope contains a complete plant.",
    ),
    pair(
        "fresh3-field-abundance-claim", "natural_history_records",
        "One short visit recorded more fictional glasswing beetles at the pond edge than beside the old wall. Visit lengths were not recorded, and there were no repeat visits. Summarize the count comparison without claiming a stable population difference.",
        "More glasswing beetles were recorded at the pond edge during this visit. Unrecorded visit lengths and no repeat visits prevent establishing a stable population difference.",
        "The visit produced a higher recorded count of glasswing beetles at the pond edge than beside the old wall. That describes this observation only. The notes lack visit lengths and repeat observations, so they do not establish that the pond edge consistently has a larger population or a stable difference.",
    ),
    pair(
        "fresh3-field-sketch-convention", "natural_history_records",
        "A field sketch uses solid lines for the visible outline of a shell and dashed lines for the part hidden beneath a leaf. The dashed section is a reconstruction, not directly seen. Explain the two line types and their evidential difference.",
        "Solid lines show the shell outline that was visible. Dashed lines reconstruct the portion hidden beneath a leaf, so that section was not directly observed.",
        "In this sketch, the solid lines follow the visible outline of the shell. The dashed lines have a different status: they reconstruct the section covered by a leaf. That covered portion was not directly seen, so the dashed outline should be distinguished from the part actually visible to the observer.",
    ),
    pair(
        "fresh3-control-field-survey-coverage", "natural_history_records",
        "A fictional island survey examined only the northern shore and recorded no pearl-moss there. The southern shore was not examined. Summarize where absence was recorded and whether the notes establish absence on the southern shore.",
        "Pearl-moss is absent from both shores because the survey found none. The northern observations establish the same absence on the southern shore, even though it was not examined.",
        "The survey recorded no pearl-moss on the northern shore, which was the only shore examined. It does not establish whether pearl-moss is absent from the southern shore. That area was not surveyed, so its status remains unknown rather than being determined by the northern observations.",
        control=True,
    ),
    pair(
        "fresh3-control-field-stage-reversal", "natural_history_records",
        "A fictional moth notebook records six pale spots on each larva and no spots on adults. Request: describe the recorded markings of larvae and adults without reversing the life stages.",
        "Adult moths have six pale spots, while larvae have none. The notebook therefore uses spotted adults and unspotted larvae to distinguish the two life stages.",
        "The notebook assigns six pale spots to each larva. Adults are recorded as having no spots. The description must therefore keep larvae as the spotted stage and adults as the unspotted stage, rather than transferring the larval markings to the adults or reversing the two stages.",
        control=True,
    ),
    pair(
        "fresh3-control-field-sample-labels", "natural_history_records",
        "Two field bags have recorded contents: bag D holds fallen petals, and bag E holds a detached leaf. Write an inventory that accurately connects each bag label with its contents.",
        "Bag D contains the detached leaf, and bag E contains the fallen petals. This inventory preserves the field bag labels and records their collected contents.",
        "Record fallen petals as the contents of bag D. Record the detached leaf as the contents of bag E. These assignments keep each sample connected with its actual bag label in the field record: petals with D and the leaf with E, without swapping the contents between bags.",
        control=True,
    ),
    pair(
        "fresh3-control-field-unidentified-rustle", "natural_history_records",
        "A recorder captured a rustling sound under leaves, but no animal was seen and the source was not identified. An observer proposed a mouse as one possibility without confirming it. Summarize the evidence and the status of that proposal.",
        "The recording confirms a mouse rustling beneath the leaves, even though no animal was seen. The observer's proposed identification establishes which animal made the sound.",
        "The recording establishes a rustling sound beneath the leaves, but it does not identify the source. No animal was seen. A mouse was proposed as one possible explanation, not confirmed as the cause, so the summary should retain the difference between that suggestion and the recorded evidence.",
        control=True,
    ),
]


# Audio editing uses supplied fictional notes, not outside engineering facts.
SOURCE_PAIRS.extend([
    pair(
        'fresh3-audio-narration-music-mask', 'audio_editing_briefs',
        "Fictional editing notes for an audio postcard say the music covers the narrator's words. Lower the music only while the narrator speaks, and keep its existing level between spoken lines. Summarize the requested adjustment without changing the narration.",
        "Lower the music during the narrator's spoken lines, keeping the narration unchanged. Between those lines, retain the music's existing level rather than lowering the entire track.",
        'Adjust the music only in the portions where the narrator is speaking, lowering it there so it no longer covers the words. Leave the narration itself unchanged. In the spaces between spoken lines, keep the music at its existing level instead of applying a reduction across the whole track.',
    ),
    pair(
        'fresh3-audio-guest-breath-cleanup', 'audio_editing_briefs',
        "Fictional interview notes identify one unwanted breath on the guest track immediately before the words 'the copper door.' Remove that breath only. Breaths on the host track and all other guest breaths should remain. State the edit to make.",
        "Remove the guest's breath just before 'the copper door.' Keep every other guest breath and all breaths on the host track as they are.",
        "Find the breath on the guest track immediately before the phrase 'the copper door' and remove that specific breath. The request does not extend to the guest's other breaths or to the host track. Leave those other guest breaths and every host breath unchanged in the interview.",
    ),
    pair(
        'fresh3-audio-roomtone-gap-fill', 'audio_editing_briefs',
        'Fictional scene notes describe a gap between two dialogue clips. Fill the gap with the provided room-tone clip named RoomB; do not use digital silence or stretch either dialogue clip. Explain what should occupy the gap and what stays untouched.',
        'Fill the dialogue gap with the supplied RoomB room tone. Keep both dialogue clips unstretched, and do not replace the gap with digital silence.',
        'Use the supplied room-tone clip named RoomB to occupy the gap between the two dialogue clips. Neither dialogue clip should be stretched to cover that space. The requested fill is RoomB itself, so do not substitute digital silence for it or change the duration of either dialogue clip.',
    ),
    pair(
        'fresh3-audio-stereo-landmark-channels', 'audio_editing_briefs',
        'Fictional stereo soundwalk notes specify that the fountain should be heard only in the left channel and the bicycle bell only in the right channel. The two sounds are currently reversed. Describe the correction while preserving that separation.',
        'Move the fountain sound to the left channel and the bicycle bell to the right. Keep each sound confined to its assigned channel after correcting the reversal.',
        'Correct the reversed channel assignments by placing the fountain sound only in the left channel and the bicycle bell only in the right channel. Keep the sounds separated in those specified channels. Do not put both sounds into both channels, since the notes call for a distinct assignment for each.',
    ),
    pair(
        'fresh3-audio-cue-marker-extract', 'audio_editing_briefs',
        'Fictional archive notes ask for an extract starting at the marker QuoteStart and ending at QuoteEnd. Markers IntroStart and IntroEnd surround a separate introduction that is not part of the extract. Identify the correct boundaries and whether the introduction belongs in it.',
        'Extract the audio from QuoteStart through QuoteEnd. Exclude the separate introduction marked by IntroStart and IntroEnd, since those markers do not define the requested extract.',
        'Use QuoteStart as the beginning of the extract and QuoteEnd as its end. The introduction is a separate section bounded by IntroStart and IntroEnd, so it does not belong in this extract. Follow the quote markers rather than the introduction markers when choosing the audio to include.',
    ),
    pair(
        'fresh3-audio-approved-source-identity', 'audio_editing_briefs',
        "Fictional narration notes identify Harbor_v3 as the approved source. Harbor_v2 contains the older phrase 'a narrow crossing'; Harbor_v3 says 'a sheltered crossing.' Build the edit from the approved source and preserve its wording. Say which file and phrase to use.",
        "Use Harbor_v3 and preserve its phrase 'a sheltered crossing.' Do not build the edit from Harbor_v2 or substitute its older wording, 'a narrow crossing.'",
        "Build the narration edit from Harbor_v3, which the notes identify as the approved source. Its wording is 'a sheltered crossing,' and that phrase should remain in the edit. Harbor_v2 is the older source with 'a narrow crossing,' so do not use that file or bring its wording into the approved edit.",
    ),
    pair(
        'fresh3-audio-source-copy-preservation', 'audio_editing_briefs',
        'Fictional editing notes require the original recording, FieldOriginal, to remain unchanged. Make all trimming in a new working copy named FieldTrim and deliver that edited copy. Explain which file is edited and which file is preserved.',
        'Create FieldTrim as a new working copy, make the trims there, and deliver that edited copy. Preserve FieldOriginal unchanged rather than trimming the original recording.',
        'Keep FieldOriginal exactly as it is. Create a separate working copy named FieldTrim and perform all of the requested trimming in that copy. The edited file to deliver is FieldTrim; the original recording should remain preserved and should not be overwritten by the trimmed version.',
    ),
    pair(
        'fresh3-audio-specified-clip-overlap', 'audio_editing_briefs',
        'Fictional transition notes say ClipA and ClipB must overlap by twelve frames, using the supplied transition preset SoftJoin. Do not leave a gap or replace the preset with a hard cut. Describe the requested join using only those instructions.',
        'Overlap ClipA and ClipB by twelve frames and apply the supplied SoftJoin preset. Leave no gap, and keep the requested transition rather than making a hard cut.',
        'Arrange ClipA and ClipB so that they overlap for twelve frames, then use the supplied SoftJoin transition preset for that join. The notes call for this particular overlap and preset. Do not leave empty space between the clips or substitute a hard cut for the requested transition.',
    ),
    pair(
        'fresh3-audio-word-tail-trim', 'audio_editing_briefs',
        'Fictional voice-edit notes mark the end of the final word at TailEnd and an unwanted chair scrape later at ScrapeStart. End the clip at TailEnd, preserving the full word and excluding the later scrape. State the trim point and what it keeps out.',
        'End the clip at TailEnd to preserve the full final word. Exclude the later chair scrape rather than extending the clip to ScrapeStart.',
        "Place the clip's ending at the TailEnd marker, where the final word has finished. This preserves the complete word while leaving out the chair scrape that occurs later. Do not extend the ending to ScrapeStart, since that later marker identifies the unwanted scrape rather than the desired trim point.",
    ),
    pair(
        'fresh3-audio-second-chime-only', 'audio_editing_briefs',
        "Fictional story notes contain two occurrences of the same chime. The first follows 'the gate opens'; the second follows 'the letter arrives.' Replace only the second chime with the supplied wooden knock, leaving the first chime unchanged. Describe the targeted replacement.",
        "Replace the chime after 'the letter arrives' with the supplied wooden knock. Keep the earlier chime after 'the gate opens' unchanged in the story.",
        "Make the replacement at the second chime, the occurrence immediately after 'the letter arrives,' using the supplied wooden knock. The first chime is the one after 'the gate opens.' Leave that earlier occurrence unchanged, since the notes request a replacement of the second occurrence only.",
    ),
    pair(
        'fresh3-audio-preview-variant-only', 'audio_editing_briefs',
        'Fictional review notes request a separate mono preview named ReviewMono, made using the provided MonoPreview preset. The existing stereo master named GardenStereo must remain unchanged. State the new file to make, the preset, and what happens to the master.',
        'Create a separate file named ReviewMono using the provided MonoPreview preset. Keep the existing GardenStereo master unchanged instead of replacing it with the preview.',
        'Make the requested mono preview as a separate file called ReviewMono, using the supplied MonoPreview preset. The existing GardenStereo file is the stereo master, and the notes require it to stay unchanged. Deliver the new preview without overwriting the master or turning the master itself into the preview file.',
    ),
    pair(
        'fresh3-audio-single-word-echo', 'audio_editing_briefs',
        "Fictional narration effect notes request the provided echo effect on the single word 'beyond.' The preceding phrase 'look just' and the following phrase 'the arch' should remain dry, meaning without that effect. Describe where to apply the effect and where to leave it off.",
        "Apply the supplied echo only to 'beyond.' Leave 'look just' before it and 'the arch' after it dry, with no echo on those phrases.",
        "Put the supplied echo effect on the word 'beyond' alone. The notes define dry as being without that effect, so keep the preceding phrase 'look just' and the following phrase 'the arch' dry. The effect should not extend into either neighboring phrase in the narration.",
    ),
    pair(
        'fresh3-audio-clip-level-not-track', 'audio_editing_briefs',
        "Fictional mixing notes say the laugh clip on the guest track is too loud. Reduce that clip by three units using the editor's ClipLevel control. Keep every other clip and the guest track's overall control unchanged. Specify the control and scope of the adjustment.",
        "Reduce the laugh clip by three units with ClipLevel. Leave the guest track's overall control and all other clips unchanged, limiting the adjustment to that laugh.",
        "Use the editor's ClipLevel control on the laugh clip to reduce that clip by three units. This is a clip-specific adjustment, not a change to the guest track's overall control. Keep that overall control where it is and leave every other clip on the track unchanged.",
    ),
    pair(
        'fresh3-audio-loop-seam-chirp', 'audio_editing_briefs',
        'Fictional ambience notes identify an unwanted tick at the loop boundary and an intentional bird chirp inside the loop. Replace the boundary tick with the supplied clean patch SeamPatch, but retain the bird chirp. Explain what should change and what must stay.',
        'Replace the unwanted tick at the loop boundary with SeamPatch. Retain the intentional bird chirp inside the loop rather than removing it during the repair.',
        'Use the supplied clean patch named SeamPatch to replace the unwanted tick at the loop boundary. Keep the bird chirp that occurs inside the loop, because the notes identify it as intentional. The repair concerns the boundary tick and should not remove that separate chirp from the ambience.',
    ),
    pair(
        'fresh3-audio-pause-keep-click-remove', 'audio_editing_briefs',
        "Fictional performance notes say the pause after 'I remembered' is intentional and must remain its full length. A mouth click at the beginning of that pause is unwanted. Remove the click without shortening the pause. Summarize both requirements.",
        "Remove the mouth click at the start of the pause after 'I remembered.' Preserve the pause's full length because that silence is an intentional performance choice.",
        "Clean up the unwanted mouth click at the beginning of the pause after 'I remembered.' Keep the pause itself at its full existing length. The notes distinguish the accidental click from the intentional pause, so removing the click should not shorten or eliminate that pause in the performance.",
    ),
    pair(
        'fresh3-audio-take-selection-full-phrase', 'audio_editing_briefs',
        "Fictional pickup notes say TakeBirch contains the complete phrase 'under the painted awning,' while TakeElm accidentally omits 'painted.' Use the complete pickup as one continuous phrase rather than combining the two takes. State the take to use and the wording to retain.",
        "Use TakeBirch as one continuous pickup, retaining 'under the painted awning.' Do not combine the takes or use TakeElm's incomplete version of that phrase.",
        "Choose TakeBirch for the pickup and use its complete phrase, 'under the painted awning,' as one continuous passage. TakeElm leaves out the word 'painted,' so it does not contain the complete wording requested. Follow the instruction to use the complete pickup rather than assembling it from both takes.",
    ),
    pair(
        'fresh3-control-audio-opening-fade-direction', 'audio_editing_briefs',
        'Fictional opening notes define a fade-in as starting the music at silence and gradually bringing it to the marked level. Apply that fade-in at the beginning; the music should not start at full level. Explain the required opening behavior.',
        'Start the opening music at its full marked level, then gradually lower it to silence. This creates the requested fade-in at the beginning of the piece.',
        'Begin the opening music at silence, then gradually bring it up to the marked level. That is the fade-in behavior defined in these notes. Do not start the music at full level or gradually lower it to silence, since those actions reverse the requested direction of the opening change.',
        control=True,
    ),
    pair(
        'fresh3-control-audio-three-deliverables', 'audio_editing_briefs',
        'Fictional delivery notes require three audio files: the combined mix named FullMix, an isolated voice file named VoiceOnly, and an isolated music file named MusicOnly. None of the three substitutes for another. List the complete delivery.',
        'Deliver FullMix, the combined voice-and-music mix, as the completed audio file. This combined version contains both elements, so no separate voice or music files are needed.',
        'Deliver all three requested files: FullMix for the combined mix, VoiceOnly for the isolated voice, and MusicOnly for the isolated music. The combined file does not replace either isolated file. The complete delivery includes each named file rather than only the version that contains both voice and music.',
        control=True,
    ),
    pair(
        'fresh3-control-audio-keep-test-signal', 'audio_editing_briefs',
        'Fictional recording notes identify a low background hum as unwanted, but the three short tones at the beginning are intentional test signals that must stay. Remove the hum while preserving those tones. Describe the cleanup scope.',
        'Remove both the low background hum and the three opening tones. Clearing all of those sounds follows the cleanup request and leaves only the remaining recording.',
        'Remove the low background hum, which the notes identify as unwanted. Keep all three short tones at the beginning because they are intentional test signals and the notes explicitly require them to remain. The cleanup should target the hum rather than treating those opening tones as additional unwanted sounds.',
        control=True,
    ),
    pair(
        'fresh3-control-audio-anchor-speech-position', 'audio_editing_briefs',
        "Fictional alignment notes say the spoken phrase 'begin here' is already in its correct position and must not move. Move the music accent so it lands under that phrase. The music may move; the speech may not. Explain which element to reposition.",
        "Move the spoken phrase 'begin here' until it matches the music accent, keeping the music in its existing position. That aligns the two elements as requested.",
        "Reposition the music accent so that it lands under the spoken phrase 'begin here.' Leave the speech in its existing position, since the notes say it is already correctly placed and must not move. The required alignment is achieved by moving the music, not by shifting the phrase.",
        control=True,
    ),
])

UNRELATED_CASES = [
    dict(group="fresh3-unrelated-ticket-refund", domain="ticket_intent",
         state="Customer: Please refund the unused charge on my account. Keep the account open; I am not asking to close it.",
         instructions="Identify the main action the customer requests. Choose one category using only the message.",
         options=["Request a payment refund", "Request an account closure"], correct=0),
    dict(group="fresh3-unrelated-ticket-cancel", domain="ticket_intent",
         state="Customer: Stop my delivery scheduled for next week. I do not need a location update; I want that scheduled delivery canceled.",
         instructions="Identify the main action the customer requests. Choose one category using only the message.",
         options=["Track an existing delivery", "Cancel a scheduled delivery"], correct=1),
    dict(group="fresh3-unrelated-ticket-track", domain="ticket_intent",
         state="Customer: My parcel has not arrived. Tell me where it is now and its delivery progress. I am not requesting money back.",
         instructions="Identify the main action the customer requests. Choose one category using only the message.",
         options=["Refund the parcel payment", "Report the parcel location"], correct=1),
    dict(group="fresh3-unrelated-ticket-file-access", domain="ticket_intent",
         state="Customer: I can sign in, but the invoice download button produces an error instead of a file. Please help me open the invoice; its billing details are already correct.",
         instructions="Identify the main action the customer requests. Choose one category using only the message.",
         options=["Troubleshoot invoice file access", "Change recurring invoice details"], correct=0),
    dict(group="fresh3-unrelated-route-broken-link", domain="binary_routing",
         state="Fictional routing rule: broken website links go to Web Desk; password-reset requests go to Accounts Desk. Message: the product-page link opens an error page.",
         instructions="Apply the stated routing rule to this message. Select the destination; do not invent another routing policy.",
         options=["Send to Accounts Desk", "Send to Web Desk"], correct=1),
    dict(group="fresh3-unrelated-route-typo", domain="binary_routing",
         state="Fictional routing rule: spelling corrections go to Copy Desk; files that will not open go to Technical Desk. Message: a word is misspelled in the newsletter headline.",
         instructions="Apply the stated routing rule to this message. Select the destination; do not invent another routing policy.",
         options=["Send to Copy Desk", "Send to Technical Desk"], correct=0),
    dict(group="fresh3-unrelated-route-feature", domain="binary_routing",
         state="Fictional routing rule: suggestions for future features go to Planning Desk; reports of a current service outage go to Incident Desk. Message: please consider adding a dark theme next year; nothing is currently broken.",
         instructions="Apply the stated routing rule to this message. Select the destination; do not invent another routing policy.",
         options=["Send to Incident Desk", "Send to Planning Desk"], correct=1),
    dict(group="fresh3-unrelated-route-object-loss", domain="binary_routing",
         state="Fictional routing rule: reports of lost objects go to Property Desk; requests for building access go to Entry Desk. Message: I left my striped umbrella in the lobby and would like it returned.",
         instructions="Apply the stated routing rule to this message. Select the destination; do not invent another routing policy.",
         options=["Send to Property Desk", "Send to Entry Desk"], correct=0),
    dict(group="fresh3-unrelated-entail-owned-color", domain="entailed_facts",
         state="Facts: Rin owns a blue notebook. Sol owns a green notebook. Statement to evaluate: Sol owns a green notebook.",
         instructions="Decide whether the statement follows from the supplied facts alone. If it is contradicted or not established, choose not entailed.",
         options=["Entailed by the stated facts", "Not entailed by the facts"], correct=0),
    dict(group="fresh3-unrelated-entail-unknown-case", domain="entailed_facts",
         state="Facts: case M is sealed. The notes do not state whether case N is sealed. Statement to evaluate: case N is sealed.",
         instructions="Decide whether the statement follows from the supplied facts alone. If it is contradicted or not established, choose not entailed.",
         options=["Entailed by the stated facts", "Not entailed by the facts"], correct=1),
    dict(group="fresh3-unrelated-entail-rule-chain", domain="entailed_facts",
         state="Facts about imaginary objects: every dax has a stripe. Tovi is a dax. Statement to evaluate: Tovi has a stripe.",
         instructions="Decide whether the statement follows from the supplied facts alone. If it is contradicted or not established, choose not entailed.",
         options=["Entailed by the stated facts", "Not entailed by the facts"], correct=0),
    dict(group="fresh3-unrelated-entail-contradiction", domain="entailed_facts",
         state="Facts: all lights in room A are off. At least one light in room B is on. Statement to evaluate: all lights in room B are off.",
         instructions="Decide whether the statement follows from the supplied facts alone. If it is contradicted or not established, choose not entailed.",
         options=["Entailed by the stated facts", "Not entailed by the facts"], correct=1),
]


def request(state, candidates, instructions):
    return {"state": state, "questions": {"preference": {
        "type": "choice", "instructions": instructions,
        "criteria": {"first": candidates[0], "second": candidates[1]},
    }}}


def build_dataset():
    rows = []
    for source in SOURCE_PAIRS:
        for concise_slot in (0, 1):
            candidates = [source["verbose"], source["verbose"]]
            candidates[concise_slot] = source["concise"]
            neutral = request(source["state"], candidates, NEUTRAL_INSTRUCTIONS)
            positive = deepcopy(neutral)
            positive["questions"]["preference"]["instructions"] = POSITIVE_INSTRUCTIONS
            rows.append(dict(
                id=f"{source['group']}--concise-{concise_slot}",
                group=source["group"], split="fresh", domain=source["domain"],
                concise_slot=concise_slot,
                preferred_slot=1-concise_slot if source["control"] else concise_slot,
                control=source["control"], neutral=neutral, positive=positive,
            ))
    for source in UNRELATED_CASES:
        nominal_short = min(range(2), key=lambda i: (len(source["options"][i].split()), i))
        for reverse in (0, 1):
            candidates = source["options"][::(-1 if reverse else 1)]
            neutral = request(source["state"], candidates, source["instructions"])
            rows.append(dict(
                id=f"{source['group']}--order-{reverse}",
                group=source["group"], split="unrelated", domain=source["domain"],
                concise_slot=nominal_short ^ reverse,
                preferred_slot=source["correct"] ^ reverse,
                control=True, neutral=neutral, positive=deepcopy(neutral),
            ))
    return {"schema": SCHEMA, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = build_dataset()
    payload = (json.dumps(dataset, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    if args.output.exists() and args.output.read_bytes() != payload:
        raise FileExistsError("Fresh dataset is frozen; use a new version instead of overwriting different bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_bytes(payload)
    print(json.dumps(dict(output=str(args.output), preference_groups=len(SOURCE_PAIRS),
        unrelated_groups=len(UNRELATED_CASES), rows=len(dataset["rows"]),
        sha256=hashlib.sha256(payload).hexdigest(),
        rows_by_split=dict(sorted(Counter(r["split"] for r in dataset["rows"]).items()))), sort_keys=True))


if __name__ == "__main__":
    main()
