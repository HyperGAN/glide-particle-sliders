#!/usr/bin/env python3
"""Build a small, original synthetic GLiDE preference-distillation dataset.

This is an instructor-defined concision preference, not a real user's learned
preference. Main pairs intentionally hold requested factual content constant;
quality controls instead make the shorter reply wrong or incomplete. The data
remain synthetic, manually authored, and too small to establish broad transfer.
Each scenario is authored independently rather than expanded from a template.
Scenario/task families stay within one split, and both option orders share one
group. The two held-out domains appear only in test.

Run from any directory; --output defaults to data/concise-preference-v1.json.
Only the question instructions change between the neutral and positive request.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "concise-preference-v1.json"


def pair(group, split, domain, state, concise, verbose, *, control=False):
    """Internal source record; row construction assigns opaque option keys."""
    return {
        "group": group,
        "split": split,
        "domain": domain,
        "state": state,
        "concise": concise,
        "verbose": verbose,
        "control": control,
    }


# Each item is a distinct scenario/task family. In particular, dev tasks are not
# numeric/name substitutions of train scenarios from the same domain.
SOURCE_PAIRS = [
    pair(
        "coding-active-filter", "train", "coding_explanations",
        "A function returns [row['id'] for row in rows if row['active']]. It does not modify rows. Request: explain what the function returns and whether it changes the input.",
        "It returns the IDs of active rows in their original order. The function creates a new list and leaves the input rows unchanged.",
        "The result is a new list containing an ID for each row whose active flag is true. Those IDs follow the same order as the original rows. Nothing in this operation changes the rows supplied to the function; the input remains as it was.",
    ),
    pair(
        "coding-cache-policy", "train", "coding_explanations",
        "The fictional Pine service caches successful results separately for each workspace for 180 seconds. Failed requests are never cached. Request: explain its cache scope, lifetime, and treatment of failures.",
        "Successful results are cached per workspace for 180 seconds. Each workspace has its own cache, and failed requests are never stored in it.",
        "Pine keeps a separate cache for each workspace rather than sharing entries across workspaces. A successful result remains cached for 180 seconds. Requests that fail do not create cached entries, so the policy stores successful results only and excludes failures entirely.",
    ),
    pair(
        "coding-import-button", "train", "coding_explanations",
        "An import button stays disabled until a file is selected and a delimiter is chosen. Both are required; neither has a default. Request: write a helpful explanation for someone seeing the disabled button.",
        "Select a file and choose a delimiter to enable Import. Both selections are required, so the button stays disabled until you have provided them.",
        "Import becomes available once you have selected the file you want to use and chosen its delimiter. Neither setting is supplied automatically. If the button is still disabled, check that both selections have been made, since completing just one of them will not enable it.",
    ),
    pair(
        "coding-retry-change", "train", "coding_explanations",
        "A patch raises the maximum retry count from two to four. The three-second wait between retries is unchanged. Request: summarize exactly what changed and what stayed the same.",
        "The patch allows up to four retries instead of two. It keeps the existing three-second wait between retries, so only the retry limit changes.",
        "With this patch, an operation can be retried as many as four times, whereas the previous limit was two retries. The timing setting has not been altered: there is still a three-second wait between retries. The change therefore affects the limit, while leaving the interval alone.",
    ),
    pair(
        "coding-upload-size-rule", "train", "coding_explanations",
        "A fictional upload validator accepts files of at most 12 megabytes, including exactly 12. Larger files are rejected before upload begins. Request: describe the size rule and when rejection happens.",
        "Files up to and including 12 megabytes are accepted. A file larger than that is rejected by validation before any upload begins.",
        "The validator permits a file when its size is 12 megabytes or less; a file of exactly 12 megabytes is within the limit. Anything above that size fails validation. That rejection occurs before uploading starts, rather than after the file has already begun to upload.",
    ),
    pair(
        "coding-log-severity", "train", "coding_explanations",
        "A worker writes DEBUG when it schedules a retry, INFO after a successful job, and WARN after the final failed attempt. Request: explain the meaning of these three worker log levels.",
        "DEBUG marks a scheduled retry, INFO records a successful job, and WARN means the final attempt failed. These levels distinguish retries, success, and exhausted attempts.",
        "For this worker, a DEBUG entry tells you that another attempt has been scheduled. An INFO entry reports that the job succeeded. A WARN entry appears when the last available attempt has failed, so it represents an exhausted set of attempts rather than an ordinary retry.",
    ),
    pair(
        "coding-name-migration", "dev", "coding_explanations",
        "A database migration combines given_name and family_name into display_name with one space between nonempty parts. A missing part is omitted, and both missing parts produce an empty string. Request: describe the mapping, including missing values.",
        "Build display_name by joining nonempty name parts with one space. Omit missing parts; if neither part is present, store an empty string.",
        "The migration takes the available given and family names and places a single space between them when both are present. It skips a missing part instead of adding an extra space for it. When both parts are missing, display_name becomes an empty string.",
    ),
    pair(
        "coding-draft-publication", "dev", "coding_explanations",
        "In the fictional Pebble editor, Save writes a private draft. Publish exposes the most recently saved draft. Unsaved edits never appear on the public page. Request: explain how to put the current edits online.",
        "Save the current edits, then choose Publish. Publish exposes the latest saved draft, while unsaved edits stay off the public page until saved.",
        "First use Save so that the current edits become the private saved draft. Then choose Publish to expose that draft on the public page. Publishing alone does not include unsaved work: it uses the most recent saved version, which is why saving needs to come first.",
    ),
    pair(
        "support-parcel-address", "train", "customer_support",
        "A parcel has a label but has not been dispatched. The store can still request an address change through its order form; the customer receives confirmation once the carrier accepts it. Request: explain the next step without claiming the change is already confirmed.",
        "Submit the new address through the order form while the parcel is still undispatched. You will receive confirmation once the carrier accepts the change.",
        "Because the parcel has not yet been dispatched, you can submit an address-change request using the order form. Enter the new address there to start the request. The change is not confirmed yet; confirmation will arrive after the carrier has accepted the requested update.",
    ),
    pair(
        "support-workshop-cancellation", "train", "customer_support",
        "A fictional workshop allows free cancellation until noon on Thursday. After that, the booking can be transferred to another person but cannot be refunded. Request: explain the two options and the cutoff.",
        "Cancel by noon Thursday for a free cancellation. After that cutoff, you can transfer the workshop booking to someone else, but no refund is available.",
        "You have until noon on Thursday to cancel without a charge. Once that deadline has passed, the workshop booking is no longer refundable. There is still another option after the cutoff: you may transfer the booking to a different person who can attend in your place.",
    ),
    pair(
        "support-pending-charge", "train", "customer_support",
        "The store's record shows one settled payment and one pending authorization for order F-218. The pending authorization is not a second settled payment. Request: explain this status without promising a release date.",
        "Order F-218 has one settled payment and one pending authorization, not two settled payments. The record does not provide a release date for the authorization.",
        "The payment record lists one completed, settled payment for order F-218. The other entry is a pending authorization rather than another settled payment, so the record does not show two completed charges. It also does not state when that authorization will be released.",
    ),
    pair(
        "support-subscription-pause", "train", "customer_support",
        "A fictional subscription pause begins at the next renewal. Current paid access continues until then, and the pause does not refund the current period. Request: explain when pausing takes effect and what happens to current access and payment.",
        "The pause starts at your next renewal. Your current paid access continues until then, and pausing does not refund the period you have already paid for.",
        "Requesting a pause changes what happens at your next renewal; it does not stop the current period immediately. You can keep using the access already paid for until that point. The payment for this current period is not refunded as part of the pause.",
    ),
    pair(
        "support-museum-timeslot", "train", "customer_support",
        "A museum can exchange an unused ticket for another time on the same day if space is available. The customer must show the original ticket at the desk. Request: explain the conditions and how to request the exchange.",
        "Bring your unused original ticket to the desk. Staff can exchange it for another time that same day if the requested slot has space.",
        "You can ask for a time-slot exchange by taking the original ticket to the museum desk. The ticket must still be unused, and the replacement time must be on the same day. Staff can make the exchange only if there is space in that time slot.",
    ),
    pair(
        "support-library-card", "train", "customer_support",
        "A fictional library replaces a lost membership card at its service desk. A photo ID is required, replacement is free, and the membership number stays the same. Request: explain what to bring, where to go, and what changes.",
        "Bring photo ID to the library service desk for a free replacement card. Your membership number stays the same, so only the physical card changes.",
        "Visit the library service desk with your photo ID to request a replacement for the lost card. There is no replacement fee. The library keeps your existing membership number rather than assigning a new one, so the replacement changes the card itself but not that number.",
    ),
    pair(
        "support-rental-damage-report", "dev", "customer_support",
        "A rented projector arrived with a cracked casing. The rental team asks for a photo of the crack and the serial number before arranging a replacement. They say not to ship anything back yet. Request: explain what the customer should do now.",
        "Send the rental team a photo of the cracked casing and the projector's serial number. Keep the projector for now; do not ship it back yet.",
        "Please send the team two things: a photo showing the crack in the casing and the projector's serial number. They need those details before arranging the replacement. For the moment, keep the projector with you and wait for further instructions rather than sending it back.",
    ),
    pair(
        "support-device-verification", "dev", "customer_support",
        "A fictional account service has paused sign-in from a new device. The account holder must enter the verification code sent to the recovery email on the sign-in screen. Support never asks the holder to send the code in a message. Request: explain the next step.",
        "Enter the code from your recovery email on the sign-in screen to verify the new device. Do not send that code to support in a message.",
        "To continue signing in on the new device, check the account's recovery email for the verification code. Enter it directly on the sign-in screen. You do not need to send the code to a support agent; support does not ask you to share it in a message.",
    ),
    pair(
        "household-chore-roster", "train", "household_planning",
        "For Saturday's chores, Mina can vacuum, Leo can wash dishes, and Sol can water the balcony plants. Each person should receive exactly that one task. Request: write the roster clearly without adding jobs.",
        "Saturday's roster is Mina for vacuuming, Leo for washing dishes, and Sol for watering the balcony plants. Each person has exactly one assigned task.",
        "For Saturday, assign the vacuuming to Mina. Leo should take responsibility for washing the dishes, while Sol should water the plants on the balcony. This roster gives each of the three people one task and includes no jobs beyond the ones listed in the plan.",
    ),
    pair(
        "household-recipe-doubling", "train", "household_planning",
        "A recipe for two servings uses one cup of lentils, three cups of stock, and two carrots. Request: list the ingredient amounts for four servings, keeping the proportions unchanged.",
        "For four servings, use two cups of lentils, six cups of stock, and four carrots. Doubling each ingredient keeps the original recipe's proportions unchanged.",
        "Four servings is twice the original two-serving recipe, so each listed ingredient needs to be doubled. That means using two cups of lentils instead of one, six cups of stock instead of three, and four carrots instead of two. These amounts preserve the original proportions.",
    ),
    pair(
        "household-bookshelf-boxes", "train", "household_planning",
        "The moving plan assigns hardbacks to the blue box, paperbacks to the yellow box, and notebooks to the green box. Request: write a packing reminder that includes all three categories and box colors.",
        "Pack hardbacks in the blue box, paperbacks in the yellow box, and notebooks in the green box. Keep each category in its assigned box.",
        "When packing the shelf, use the blue box for the hardback books. Put paperback books into the yellow box, and place notebooks in the green one. Following those three assignments keeps every listed category in the box color specified by the moving plan.",
    ),
    pair(
        "household-wrap-sorting", "train", "household_planning",
        "After a party, the household wants uncreased wrapping paper stored flat, ribbons rolled into a tin, and torn paper put in the craft drawer. Request: explain the sorting plan without discarding anything.",
        "Store uncreased wrapping paper flat, roll the ribbons into a tin, and put torn paper in the craft drawer. Keep all three materials for reuse.",
        "Separate the leftover materials according to their condition and type. Wrapping paper without creases should be stored flat. Roll the ribbons and place them in a tin, then put the torn paper in the craft drawer. Nothing in this sorting plan needs to be thrown away.",
    ),
    pair(
        "household-errand-order", "train", "household_planning",
        "A fictional town map puts the post office directly beside home, the market next, and the repair shop beyond the market. All are on one road, and the household wants to visit them in that outward order. Request: state the errand sequence.",
        "Start at home, then visit the post office, the market, and the repair shop in that order. This follows the requested outward route along the road.",
        "Leave home and make the post office your first stop, since it is directly beside home on the supplied map. Continue along the same road to the market. After the market, go on to the repair shop, completing the stops in the outward order requested.",
    ),
    pair(
        "household-watering-rota", "train", "household_planning",
        "The household's written plant rota assigns the fern to Monday, the ivy to Wednesday, and the palm to Friday. Request: repeat the rota exactly, without adding care advice or changing days.",
        "Follow the written rota: water the fern on Monday, the ivy on Wednesday, and the palm on Friday. Those are the three scheduled assignments.",
        "The rota sets aside Monday for the fern. Wednesday is the scheduled day for the ivy, and Friday is the scheduled day for the palm. Keeping those assignments as written preserves the household's existing plan without introducing any additional tasks or different watering days.",
    ),
    pair(
        "household-donation-catalog", "dev", "household_planning",
        "A donation catalog should list a red scarf under clothing, a ceramic bowl under kitchenware, and a wooden puzzle under toys. Request: create the three catalog entries with their categories.",
        "Catalog the red scarf as clothing, the ceramic bowl as kitchenware, and the wooden puzzle as toys. These entries cover all three donated items.",
        "Add an entry for the red scarf in the clothing category. The ceramic bowl belongs in the kitchenware category, and the wooden puzzle belongs in toys. The resulting catalog has one entry for each donated item, using the three categories specified in the instructions.",
    ),
    pair(
        "household-room-setup", "dev", "household_planning",
        "For a board-game gathering, the room plan places the game table by the window, snacks on the sideboard, and spare chairs against the back wall. Request: describe the layout so a helper can set it up.",
        "Put the game table by the window, arrange snacks on the sideboard, and place spare chairs against the back wall. This matches the gathering's room plan.",
        "Set the game table beside the window as the room plan specifies. Use the sideboard as the place for the snacks. Finally, position any spare chairs against the back wall. Those three placements reproduce the intended layout for the board-game gathering and give the helper each assigned location.",
    ),
    pair(
        "education-water-cycle", "train", "educational_explanations",
        "Class notes say evaporation changes liquid water into vapor, condensation changes vapor into droplets, and precipitation returns water from clouds to the ground. Request: summarize these three stages using only the notes.",
        "Evaporation turns liquid water into vapor, condensation turns vapor into droplets, and precipitation carries water from clouds back to the ground. These are the three stages described.",
        "The notes begin with evaporation, in which liquid water becomes vapor. During condensation, that vapor changes into droplets. Precipitation is the stage that brings water from clouds down to the ground. Together, these descriptions summarize the three named stages without adding any other processes.",
    ),
    pair(
        "education-fraction-equivalence", "train", "educational_explanations",
        "A worksheet shows one half equals two quarters because multiplying both numerator and denominator of 1/2 by two gives 2/4. Request: explain the equality using that method.",
        "Multiply both parts of one half by two: one becomes two and two becomes four. This gives two quarters without changing the fraction's value.",
        "Start with the fraction one half. Multiplying its numerator by two changes one to two, and multiplying its denominator by the same number changes two to four. The result is two quarters. Because both parts were multiplied by two, the fraction keeps the same value.",
    ),
    pair(
        "education-character-inference", "train", "educational_explanations",
        "A story says Neri finds a lost sketchbook, sees its owner's name, and takes it to the school office instead of keeping it. Request: describe what this action suggests about Neri and cite the action as evidence.",
        "Neri seems considerate and honest: after finding the named sketchbook, Neri takes it to the school office rather than keeping something that belongs to another person.",
        "Neri's action suggests a considerate, honest character. The evidence is what happens after the sketchbook is found: Neri notices the owner's name and takes the book to the school office. Choosing to turn it in instead of keeping it supports that interpretation of the character.",
    ),
    pair(
        "education-experiment-variables", "train", "educational_explanations",
        "Students compare paper gliders by changing only wing width and measuring flight distance. Paper type, launch height, and launcher stay the same. Request: identify the changed variable, measured result, and three controlled conditions.",
        "Wing width is changed, and flight distance is measured. Paper type, launch height, and the launcher are kept constant so those conditions do not vary.",
        "The students deliberately vary wing width, making that the changed variable. Flight distance is the result they measure for each glider. To keep the comparison focused on wing width, they hold three conditions constant: the type of paper, the launch height, and the person launching.",
    ),
    pair(
        "education-metaphor-definition", "train", "educational_explanations",
        "The lesson defines a metaphor as describing one thing as another to suggest a shared quality. Its example, 'the classroom was a beehive,' suggests busy activity. Request: explain the example using this definition.",
        "Calling the classroom a beehive is a metaphor that suggests busy activity. It describes the classroom as another thing to highlight a shared quality.",
        "In the example, the classroom is described as a beehive rather than merely compared with one. The shared quality supplied by the lesson is busy activity. This makes the sentence a metaphor under the given definition: one thing is described as another to suggest that quality.",
    ),
    pair(
        "education-table-change", "train", "educational_explanations",
        "A practice table gives a tank level of 18 units at minute zero and 30 units at minute four. Request: calculate the total increase and the average increase per minute over those four minutes.",
        "The level rises by 12 units, from 18 to 30. Dividing that increase by four minutes gives an average increase of three units per minute.",
        "Subtract the starting level of 18 units from the ending level of 30 units to find a total increase of 12 units. That change occurs across four minutes. Dividing 12 by four gives the average rate over the interval: three units of increase per minute.",
    ),
    pair(
        "education-counterexample", "dev", "educational_explanations",
        "A classroom argument claims all festival badges are round. A student presents one triangular festival badge. Request: explain why this is a counterexample, without making claims about how common either shape is.",
        "The triangular festival badge is a counterexample because it is a festival badge that is not round. One such example disproves the claim that all are round.",
        "The claim says every festival badge has a round shape. The student supplies a festival badge with a triangular shape, so it meets the category in the claim but lacks the claimed property. That single exception is enough to contradict the word 'all,' regardless of how frequent either shape is.",
    ),
    pair(
        "education-source-attribution", "dev", "educational_explanations",
        "A class handout says direct quotations must use quotation marks and identify the author. Paraphrases use the student's own wording but still identify the author. Request: explain what the two forms share and how they differ.",
        "Both quotations and paraphrases identify the author. A quotation keeps the source's wording in quotation marks; a paraphrase uses the student's own wording instead.",
        "The shared requirement is attribution: both forms must identify the author. Their treatment of wording differs. A direct quotation preserves the source's words and places them inside quotation marks. A paraphrase expresses the material in the student's own words, while still giving the author credit.",
    ),
]


# The following domains are held out in their entirety.
SOURCE_PAIRS.extend([
    pair(
        "travel-tram-connection", "test", "travel_logistics",
        "In a fictional town, I must reach Linden Pier by 09:40. From my hotel, the tram arrives there at 09:25 and the shuttle arrives at 09:50. Both leave from outside the hotel. Tell me which service to take and why.",
        "Take the tram from outside your hotel. It reaches Linden Pier at 09:25, fifteen minutes before your deadline; the shuttle arrives too late.",
        "The tram is the service to choose from outside your hotel. Its arrival at Linden Pier is 09:25, which gives you fifteen minutes before your 09:40 deadline. The shuttle does not reach the pier until 09:50, so it would arrive after the time you need to be there.",
    ),
    pair(
        "travel-locker-clearance", "test", "travel_logistics",
        "At a fictional station, lockers close at 18:00. My bag fits the large locker but not the small locker. I will collect it at 17:30. Recommend a locker size and say whether my collection time works.",
        "Use a large locker because your bag will not fit a small one. Collecting it at 17:30 works, thirty minutes before the lockers close.",
        "Choose one of the large lockers for your bag, since the small lockers do not have enough room for it. Your planned collection time also works with the stated opening hours: 17:30 is thirty minutes earlier than the 18:00 closing time, so you can collect the bag before closure.",
    ),
    pair(
        "travel-step-free-path", "test", "travel_logistics",
        "I need a step-free walk to a fictional gallery. The east path takes eight minutes and has stairs; the west path takes twelve minutes and has a ramp throughout. Tell me which path meets my need and its walking time.",
        "Take the west path to the gallery. Its ramp makes the route step-free, and the walk takes twelve minutes; the east path includes stairs.",
        "The west path meets your requirement for a step-free walk to the gallery. It has a ramp throughout and takes twelve minutes. Although the east path is quicker at eight minutes, it includes stairs, so it does not meet the step-free requirement you gave for this walk.",
    ),
    pair(
        "travel-booking-reference", "test", "travel_logistics",
        "My fictional guesthouse says check-in requires a booking reference and photo identification. The reference is K7M2. Draft a reminder listing exactly what I need to bring, including the reference.",
        "Bring your photo identification and booking reference K7M2 when you check in at the guesthouse. Those are the two items required by its instructions.",
        "For check-in at the guesthouse, remember to bring your photo identification together with your booking reference, K7M2. The guesthouse's instructions list these two items as the requirements for checking in, so your reminder should include both the identification and the reference rather than leaving either one out.",
    ),
    pair(
        "travel-pickup-landmark", "test", "travel_logistics",
        "A fictional tour operator changed pickup from the library entrance to the blue fountain in Market Square. Departure remains 10:15. Write a message to my travel companion explaining where and when to meet.",
        "Meet me at the blue fountain in Market Square for our 10:15 departure. Pickup has moved there from the library entrance, so use the new location.",
        "Our pickup location has changed, so please meet me at the blue fountain in Market Square instead of at the library entrance. The tour operator has not changed the departure time: it is still 10:15. We should therefore use the new meeting place while keeping the same departure time.",
    ),
    pair(
        "travel-bicycle-ferry", "test", "travel_logistics",
        "Two fictional ferries leave at 11:00 and 11:30. The 11:00 ferry does not allow bicycles; the 11:30 ferry allows them. I am taking my bicycle. Tell me which departure I can use and the relevant restriction.",
        "Use the 11:30 ferry, which allows bicycles. You cannot take your bicycle on the 11:00 departure because that ferry does not permit them aboard.",
        "The 11:30 departure is the ferry you can use while taking your bicycle. Bicycles are allowed on that service. The earlier ferry leaves at 11:00, but its restriction does not allow bicycles aboard, so that departure does not meet the requirement for this journey with your bicycle.",
    ),
    pair(
        "travel-museum-sequence", "test", "travel_logistics",
        "In my fictional itinerary, the observatory visit is fixed at 14:00. The sculpture garden is next door and takes thirty minutes to visit. I can start sightseeing at 13:15. Suggest the order and a suitable garden time without changing the observatory booking.",
        "Visit the sculpture garden from 13:15 to 13:45, then go next door to the observatory for 14:00. This preserves your fixed observatory booking.",
        "Start with the sculpture garden at 13:15 and spend thirty minutes there, finishing at 13:45. After that, go next door to the observatory for your fixed 14:00 visit. This order fits the garden into your available time while leaving the observatory booking at its original time of 14:00.",
    ),
    pair(
        "travel-left-luggage-receipt", "test", "travel_logistics",
        "A fictional luggage desk requires the paper receipt to return a stored suitcase. I have receipt number 284 in my jacket pocket and plan to collect the suitcase after lunch. Write myself a reminder about the receipt and collection.",
        "After lunch, collect your suitcase from the luggage desk and bring paper receipt 284 from your jacket pocket. The desk requires that receipt for return.",
        "Remember to collect your stored suitcase from the luggage desk after lunch. Before going to the desk, take paper receipt number 284 from your jacket pocket and bring it with you. The desk requires the paper receipt before it will return the suitcase, so the receipt needs to accompany you.",
    ),
    pair(
        "workplace-release-status", "test", "workplace_communication",
        "Draft a project update using only these facts: the interface is complete, testing starts Tuesday, and the release date remains Thursday. The audience needs the current status and both dates.",
        "The interface is complete, testing starts Tuesday, and the release remains scheduled for Thursday. These are the current project status and the two upcoming dates.",
        "The current project status is that the interface work is complete. The next scheduled step is testing, which starts on Tuesday. The release date has not changed and remains Thursday. This update therefore confirms the completed interface, the Tuesday start of testing, and the continuing Thursday release schedule.",
    ),
    pair(
        "workplace-meeting-alternative", "test", "workplace_communication",
        "A colleague invited me to a planning call at 15:00. I already have another meeting then, but I am available at 16:00. Draft a polite reply declining the proposed time and offering my available time.",
        "Thanks for inviting me. I have another meeting at 15:00, so I cannot join then. Could we hold the planning call at 16:00 instead?",
        "Thank you for inviting me to the planning call. I already have another meeting at 15:00, so I am not available to join at the proposed time. I am available at 16:00, however. Would it work for you to move the planning call to that time instead?",
    ),
    pair(
        "workplace-handoff-files", "test", "workplace_communication",
        "Write a handoff note: the draft report is in the shared folder named QuarterReview, the open issue is the missing appendix, and Nia will add that appendix tomorrow. Include the file location, issue, owner, and timing.",
        "The draft report is in QuarterReview, our shared folder. Its appendix is still missing; Nia owns that remaining task and will add the appendix tomorrow.",
        "For the handoff, the draft report can be found in the shared folder named QuarterReview. The remaining open issue is that the appendix has not yet been added. Nia is responsible for that task and will add the missing appendix tomorrow, completing the specific item noted in this handoff.",
    ),
    pair(
        "workplace-decision-rationale", "test", "workplace_communication",
        "Our fictional team selected Option Cedar because it works offline. Option Maple requires a network connection. Draft a decision record stating the chosen option and the reason, without claiming any other advantage.",
        "We selected Option Cedar because it works offline. Option Maple requires a network connection, so offline operation is the reason for our team's decision.",
        "The team's decision is to select Option Cedar. The reason for this selection is its ability to work offline. Option Maple, by comparison, requires a network connection. This decision record identifies offline operation as the basis for choosing Cedar and does not claim any additional advantage for either option.",
    ),
    pair(
        "workplace-feedback-agenda", "test", "workplace_communication",
        "Write constructive feedback to a colleague. In yesterday's workshop, the discussion started before the agenda was shared. Ask them to share the agenda before discussion at the next workshop. Do not infer their intent or describe other incidents.",
        "Yesterday's workshop discussion began before the agenda was shared. For our next workshop, please share the agenda before discussion starts so everyone sees it first.",
        "In yesterday's workshop, the discussion had already started when the agenda was shared. For the next workshop, could you please share the agenda before starting the discussion? That would put the agenda in front of everyone first. My request concerns this sequence in the workshop, without making any assumption about your intent.",
    ),
    pair(
        "workplace-scope-boundary", "test", "workplace_communication",
        "Draft a reply about a fictional dashboard request. The current delivery includes weekly totals only. A new request asks for hourly totals, which require a separate estimate. State the existing scope and the next step for the new request.",
        "The current dashboard delivery includes weekly totals only. Hourly totals are an additional request, so the next step is to prepare a separate estimate for them.",
        "The scope of the current dashboard delivery covers weekly totals only. The request for hourly totals is additional to that existing scope. For the new request, the next step is to prepare a separate estimate, so that hourly totals can be considered separately from the weekly totals already included.",
    ),
    pair(
        "workplace-clarify-export", "test", "workplace_communication",
        "A colleague asked me to 'export the results' but did not specify a file format. The two available formats are CSV and PDF. Draft a clarification question that asks which format they want, without choosing for them.",
        "Which format would you like for the results export: CSV or PDF? Both are available, and I need your preference before choosing the export format.",
        "Before I export the results, could you clarify which file format you would like me to use? The available options are CSV and PDF. Your request did not specify a format, so please let me know which of those two options you want rather than having me choose one for you.",
    ),
    pair(
        "workplace-test-environment-notice", "test", "workplace_communication",
        "Write a notice about a fictional test environment. It will be unavailable Wednesday from 09:00 to 10:00 for maintenance. The production environment is unaffected. Include the affected environment, time window, reason, and production status.",
        "The test environment will be unavailable Wednesday from 09:00 to 10:00 for maintenance. This notice concerns testing only; the production environment will remain unaffected.",
        "Please note that the test environment has a maintenance window on Wednesday, starting at 09:00 and ending at 10:00. It will be unavailable during that one-hour period because of the maintenance work. The production environment is not affected by this window, so the stated unavailability applies to the test environment.",
    ),
])


# Controls intentionally make brevity conflict with required correctness or
# completeness. Their preferred reply is the longer one, under both prompts.
SOURCE_PAIRS.extend([
    pair(
        "control-coding-checkbox-export", "train", "coding_explanations",
        "A fictional table exporter writes only rows whose selection checkbox is checked. Unchecked rows are excluded. Request: explain which rows will appear in the export, preserving this selection rule.",
        "The export includes every row in the table, whether its checkbox is checked or not. Selection checkboxes do not restrict which rows appear in the exported file.",
        "The exporter includes only the rows whose selection checkboxes are checked. Any row with an unchecked checkbox is excluded from the export. Before exporting, check the boxes for the rows you want to include; the resulting file will follow those selections rather than containing every table row.",
        control=True,
    ),
    pair(
        "control-household-supply-count", "train", "household_planning",
        "Each of three display trays needs four pegs. There are no spare trays and no extra pegs required. Request: calculate the total peg count for all three trays.",
        "You need eight pegs altogether for the three display trays. That total provides the four pegs required by each tray without adding any unnecessary spares.",
        "You need twelve pegs in total. There are three display trays, and each one needs four pegs, so the calculation is three times four equals twelve. This covers all three trays exactly and does not include any spare trays or any additional pegs beyond the stated requirement.",
        control=True,
    ),
    pair(
        "control-support-order-mismatch", "dev", "customer_support",
        "A fictional support ticket says the customer ordered a blue shirt in size medium but received a red shirt in size small. Request: summarize the mismatch, accurately distinguishing what was ordered from what was received.",
        "The customer ordered a red shirt in size small but received a blue shirt in size medium. Their delivery differs from the color and size requested.",
        "The customer ordered a blue shirt in size medium, but the delivery contained a red shirt in size small. Both attributes are wrong: the received color is red rather than blue, and the received size is small rather than medium. This keeps the ordered item distinct from the delivered item.",
        control=True,
    ),
    pair(
        "control-education-shape-sides", "dev", "educational_explanations",
        "A geometry card defines a triangle as a polygon with three sides and a square as a polygon with four equal sides. Request: state the number of sides of the triangle and distinguish it from the square using the card.",
        "The triangle has four sides, just as the square does. According to this geometry card, both shapes have the same side count despite their different names.",
        "The triangle has three sides, while the square has four equal sides. The card therefore distinguishes them by side count: three for the triangle and four for the square. They do not have the same number of sides, so the triangle should not be described as four-sided.",
        control=True,
    ),
    pair(
        "control-travel-terminal-deadline", "test", "travel_logistics",
        "At a fictional airport, I must be at Terminal Two by 12:00. The green bus reaches it at 11:45; the silver bus reaches it at 12:20. Choose a bus that meets the deadline and state its arrival time.",
        "Take the silver bus to Terminal Two, arriving at 12:20. It serves the terminal you need and provides a direct choice for your airport transfer.",
        "Take the green bus to Terminal Two. It arrives at 11:45, which is fifteen minutes before your 12:00 deadline. The silver bus arrives at 12:20, so that option would reach the terminal too late. The green bus is therefore the option that meets the arrival requirement you gave.",
        control=True,
    ),
    pair(
        "control-workplace-poll-result", "test", "workplace_communication",
        "A fictional team poll selected the hexagon mark with nine votes. The circle mark received three votes and was not selected. Request: write a summary naming the selected mark and reporting both vote counts accurately.",
        "The circle mark was selected with nine votes, while the hexagon mark received three votes. This reports the chosen mark and both totals from the team's poll.",
        "The team selected the hexagon mark, which received nine votes in the poll. The circle mark received three votes and was not selected. The summary should therefore identify the hexagon as the selected mark and keep the nine-vote and three-vote totals attached to their correct options.",
        control=True,
    ),
])


def request(state, candidates, instructions):
    return {
        "state": state,
        "questions": {
            "preference": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {"first": candidates[0], "second": candidates[1]},
            }
        },
    }


def build_dataset():
    rows = []
    for source in SOURCE_PAIRS:
        for concise_slot in (0, 1):
            candidates = [source["verbose"], source["verbose"]]
            candidates[concise_slot] = source["concise"]
            neutral = request(source["state"], candidates, NEUTRAL_INSTRUCTIONS)
            positive = deepcopy(neutral)
            positive["questions"]["preference"]["instructions"] = POSITIVE_INSTRUCTIONS
            rows.append({
                "id": f"{source['group']}--concise-{concise_slot}",
                "group": source["group"],
                "split": source["split"],
                "domain": source["domain"],
                "concise_slot": concise_slot,
                "preferred_slot": 1 - concise_slot if source["control"] else concise_slot,
                "control": source["control"],
                "neutral": neutral,
                "positive": positive,
            })
    return {"schema": SCHEMA, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(row["split"] for row in dataset["rows"])
    print(json.dumps({"output": str(args.output), "groups": len(SOURCE_PAIRS), "rows": len(dataset["rows"]), "rows_by_split": dict(sorted(counts.items()))}, sort_keys=True))


if __name__ == "__main__":
    main()
