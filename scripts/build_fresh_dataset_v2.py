#!/usr/bin/env python3
"""Build blind, original fresh scenarios for a frozen concision experiment.

These examples were authored without model-output inspection or prediction-based
selection. They are synthetic tests of an instructor-defined preference, not a
real user's learned preference. Every main pair preserves the requested facts;
the shorter reply in every quality control is wrong or critically incomplete.
Both option orders belong to one scenario group. All rows use split='fresh'.

The builder is deterministic and refuses to overwrite a different existing
dataset. Final data must be hashed before encoding and never edited in response
to model performance. No old dataset, source, cache, or result file is modified.
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
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "concise-fresh-v2.json"


def pair(group, domain, state, concise, verbose, *, control=False):
    return dict(group=group, domain=domain, state=state, concise=concise,
                verbose=verbose, control=control)


SOURCE_PAIRS = [
    pair(
        "fresh-craft-collage-depth", "arts_crafts",
        "A collage sketch places a navy sheet at the back, a cream arch in front of that sheet, and a rust circle in front of the arch. Explain the three layers from back to front.",
        "Arrange the layers from back to front as navy sheet, cream arch, then rust circle. The arch covers the sheet, and the circle covers the arch.",
        "Begin with the navy sheet as the background layer. Place the cream arch in front of that sheet, so the arch sits above the background. The rust circle belongs in front of the cream arch. Reading the finished collage from back to front gives sheet, arch, and circle.",
    ),
    pair(
        "fresh-craft-woven-bookmark", "arts_crafts",
        "A bookmark design leaves both edge cords plain. Across the middle, the color motif repeats mint, mint, plum. Describe the middle motif and what happens to the two edge cords.",
        "Repeat mint, mint, plum across the bookmark's middle, while leaving both edge cords plain. The repeating color motif belongs only in the middle section.",
        "Use the three-part motif mint, mint, plum for the middle of the bookmark, repeating that sequence across the section. Keep the cord at each outer edge plain. The design separates the patterned middle from the two unpatterned edge cords rather than carrying the motif onto those edges.",
    ),
    pair(
        "fresh-craft-leaf-impression", "arts_crafts",
        "For an imaginary clay-tile exercise, the guide says to press one leaf into the soft tile, lift the leaf away, and then leave the tile to dry. Explain the exercise's sequence without adding painting or other steps.",
        "Press one leaf into the soft tile, lift the leaf away, and leave the tile to dry. Follow those three actions in that order.",
        "While the tile is still soft, press a single leaf into its surface. Next, lift the leaf away from the tile. After removing it, leave the tile to dry. Those are the actions specified by this exercise; there is no additional painting stage in the supplied guide.",
    ),
    pair(
        "fresh-craft-stencil-openings", "arts_crafts",
        "A stencil drawing marks three oval windows for removal. Its outer border and the narrow bars between the windows must remain connected. Tell the maker which parts to remove and which to retain.",
        "Remove the three oval windows. Retain the outer border and the narrow connecting bars between the windows, keeping those remaining parts joined as the drawing requires.",
        "The removable parts are the three areas marked as oval windows. Leave the outer border in place, and also retain the narrow bars separating the windows. The border and bars need to remain connected, so only the marked window areas should be taken out of this stencil.",
    ),
    pair(
        "fresh-craft-patch-rotation", "arts_crafts",
        "A patchwork correction asks for the upper-right patch to rotate one quarter-turn clockwise. Every other patch should keep its present orientation. State the correction precisely so the maker does not rotate the whole arrangement.",
        "Rotate only the upper-right patch one quarter-turn clockwise. Leave every other patch in its current orientation, and keep the overall arrangement in place.",
        "Find the patch in the upper-right position and turn that individual patch one quarter-turn in the clockwise direction. No other patch needs to turn. Keep all the other orientations as they are, and do not rotate the entire arrangement while making this single-patch correction.",
    ),
    pair(
        "fresh-craft-bead-charm", "arts_crafts",
        "The five beads of a charm run from left to right as teal, white, amber, white, teal. A small pendant hangs from the middle bead. Give the bead order and identify the bead carrying the pendant.",
        "String the beads teal, white, amber, white, teal from left to right. Hang the pendant from the amber bead in the middle of the charm.",
        "Reading from the left end of the charm, the beads should be teal, white, amber, white, and teal. This places amber as the third bead, with two beads on either side. Attach the small pendant to that central amber bead rather than to one of the white or teal beads.",
    ),
    pair(
        "fresh-craft-folded-zine", "arts_crafts",
        "The labeled panels of a folded zine are fixed: cover outside at the front, credits outside at the back, note inside left, and drawing inside right. Describe the content of the outside and inside panels.",
        "Put the cover on the outside front and credits on the outside back. Inside, place the note on the left and the drawing on the right.",
        "The outside front panel is the cover, while the outside back panel contains the credits. Open the zine to find the other two assigned panels: the note belongs on the inside left, and the drawing belongs on the inside right. Keep those four panel assignments as labeled.",
    ),
    pair(
        "fresh-craft-maker-mark", "arts_crafts",
        "A ceramic series uses a maker mark consisting of the initials LV above the collection code H3. No date is part of the mark. Describe its contents and their vertical arrangement, without inventing a date.",
        "Place the initials LV above the collection code H3. The mark contains those two elements only; a date is not included in this series design.",
        "The upper element of the maker mark is the pair of initials LV. Beneath those initials, put H3 as the collection code. These are the two elements specified for the mark. The series design does not include a date, so there is no date element to add below them.",
    ),
    pair(
        "fresh-craft-applique-edit", "arts_crafts",
        "A fabric bird already has the required blue body and gold beak. Its gray wing is the only incorrect part; the design calls for a white wing. Explain the smallest correction that makes the bird match the design.",
        "Replace the gray wing with a white wing. Keep the blue body and gold beak unchanged, because those two parts already match the design.",
        "Only the wing needs to change. Remove the gray wing and replace it with a white one, which is the color required by the design. Leave the body blue and the beak gold: both of those existing parts are already correct and need no alteration.",
    ),
    pair(
        "fresh-craft-display-cube", "arts_crafts",
        "A display cube has six faces. The project brief decorates its four vertical side faces, while its top and bottom stay blank. Explain which faces receive decoration and which do not.",
        "Decorate the cube's four vertical side faces. Leave the top and bottom faces blank, so decoration appears around the sides rather than on either horizontal face.",
        "Apply the decoration to the four faces that form the upright sides of the cube. The face at the top and the face underneath are excluded from the decoration plan. Both should stay blank, giving the cube decorated vertical sides with an undecorated top and bottom.",
    ),
    pair(
        "fresh-craft-mosaic-reflection", "arts_crafts",
        "A mosaic's left half shows an arrow pointing toward the center seam. The right half must be its mirror image. Describe the arrow direction on the right and the relationship between the two halves.",
        "The right-hand arrow should also point toward the center seam, in the opposite direction from the left arrow. The two halves mirror each other across that seam.",
        "Reflect the left half across the center seam to make the right half. Because the left arrow points inward, its reflected partner on the right also points inward toward the seam. The arrows therefore face opposite directions while forming a mirrored pair across the middle of the mosaic.",
    ),
    pair(
        "fresh-craft-lettering-proof", "arts_crafts",
        "A fictional printmaking checklist says to make a small proof first and check whether its lettering reads in the intended direction. The full run starts only after that check passes. Explain what the proof is checking and when the full run may begin.",
        "Use the small proof to check that the lettering reads in the intended direction. Begin the full run only after that lettering check passes.",
        "Make the small proof before starting the full run. Its purpose in this checklist is to check the direction in which the lettering reads. If that direction is correct, the check has passed and the full run may begin. Until then, the checklist does not permit the full run.",
    ),
    pair(
        "fresh-craft-dye-sample-choice", "arts_crafts",
        "Two dyed-paper samples are being compared with a brief that requires a muted purple without visible speckles. Sample Iris is muted purple but speckled; Sample Heather is muted purple and unspeckled. Identify the sample matching the brief and explain the difference that matters.",
        "Choose Sample Heather: it is muted purple without speckles. Sample Iris has the right color but visible speckles, which conflict with the stated brief.",
        "Sample Heather matches both parts of the brief because its purple is muted and its surface is unspeckled. Sample Iris also meets the muted-purple color requirement, but its visible speckles do not meet the surface requirement. The presence or absence of speckles is the deciding difference between these samples.",
    ),
    pair(
        "fresh-craft-texture-contrast", "arts_crafts",
        "An illustration brief wants a rough, broken texture inside the mountain shape and a thin continuous outline around it. The supplied plan uses the dry-brush tool for the interior and the fine pen for the outline. Explain how the two tools divide the work.",
        "Use the dry-brush tool for the mountain's rough interior texture and the fine pen for its thin continuous outline. Keep the two treatments in their assigned areas.",
        "Work inside the mountain shape with the dry-brush tool to create the rough, broken texture specified by the brief. Use the fine pen around the shape for the thin, continuous outline. This gives the interior and its boundary different treatments, with each tool assigned to the part described in the plan.",
    ),
    pair(
        "fresh-craft-embroidery-motif", "arts_crafts",
        "A border design has a repeating unit named curl, seed, seed. Each new unit begins with curl again; neighboring units do not share a seed. Explain the repeat so someone can continue the border correctly.",
        "Repeat curl, seed, seed as a complete unit. Start the next unit with another curl, keeping both seeds in every unit rather than sharing one between units.",
        "Treat curl, seed, seed as the full repeating unit. After its second seed, begin another complete unit with curl, followed by its own two seeds. Do not let a seed count for two neighboring units. Continuing in complete three-part units preserves the repeat given in the design.",
    ),
    pair(
        "fresh-craft-map-paper-caption", "arts_crafts",
        "A handmade lampshade uses paper cut from discarded topographic maps. The maker deliberately retained the contour lines as visible decoration. Draft a caption mentioning the material's origin and how those markings are used, without adding environmental impact claims.",
        "This lampshade uses paper from discarded topographic maps, with their contour lines retained as decoration. The visible map markings are part of the finished design.",
        "The paper in this handmade lampshade comes from discarded topographic maps. Rather than hiding the original markings, the maker kept the contour lines visible and used them as decoration. The finished design therefore displays the paper's map origins through those retained lines, without requiring any additional claim about environmental impact.",
    ),
    pair(
        "fresh-control-craft-clear-window", "arts_crafts",
        "A paper lantern design requires a clear window through which a blue circle on the inner panel remains visible. The outer panel must not cover that window with opaque paper. Explain how to preserve the intended visible circle.",
        "Cover the window with opaque paper so the blue circle is hidden. This closes the opening in the outer panel and preserves the lantern's intended design.",
        "Keep the window clear and leave the blue circle on the inner panel visible through it. Do not place opaque paper across that window in the outer panel. The design relies on seeing the circle through the clear area, so covering it would remove the intended visible element.",
        control=True,
    ),
    pair(
        "fresh-control-craft-mirrored-stems", "arts_crafts",
        "A paired ornament design has two stems that bend toward the center: the left stem bends right, and the right stem bends left. Explain the required directions for the two stems without turning the pair into identical copies.",
        "Bend both stems toward the left so the two ornaments match. The left stem and right stem should therefore be identical copies pointing in the same direction.",
        "Bend the left stem to the right and the right stem to the left. This makes both stems point inward toward the center of the pair. They should face opposite directions rather than becoming identical copies that bend the same way, since the required arrangement is inward-facing.",
        control=True,
    ),
    pair(
        "fresh-control-craft-exhibit-label", "arts_crafts",
        "An exhibition label must name the maker, material, and year. The supplied facts are maker Tess Orin, material woven paper, and year 2025. Draft a complete label using all three required details.",
        "Made by Tess Orin in 2025. This label gives the required information for the exhibition and identifies the person responsible for creating the displayed piece.",
        "Made by Tess Orin, using woven paper, in 2025. The maker is Tess Orin, the material is woven paper, and the year is 2025. These three details belong on the exhibition label because the supplied labeling requirement calls for the maker, material, and year together.",
        control=True,
    ),
    pair(
        "fresh-control-craft-movable-puppet", "arts_crafts",
        "An imaginary paper-puppet kit uses a turning paper fastener at the elbow so the forearm can move. The guide says the elbow joint must remain movable and must not be glued solid. Explain how that joint should work.",
        "Glue the elbow joint solid so the forearm cannot turn around its paper fastener. A fixed joint is the required result for this movable paper-puppet kit.",
        "Use the turning paper fastener at the elbow and leave the joint able to move. The forearm should turn at that fastener rather than being fixed in place. Do not glue the elbow solid, because the kit's guide specifically requires that joint to remain movable.",
        control=True,
    ),
]


# Fictional game mechanics form the second new domain.
SOURCE_PAIRS.extend([
    pair(
        'fresh-game-mosaic-distinct-set', 'recreation_game_rules',
        'In the fictional game Mosaic Orchard, each completed row scores two points. Having at least one completed row in each of the three patterns adds a single five-point collection bonus. I completed four rows covering all three patterns. Explain my score.',
        'Your four completed rows score eight points, and covering all three patterns adds the five-point collection bonus once. Your total is thirteen points.',
        'You score two points for each of your four completed rows, giving eight points from rows. Because those rows include all three patterns, you also qualify for the collection bonus. That bonus is awarded once and adds five points, bringing your score to thirteen points in total.',
    ),
    pair(
        'fresh-game-lantern-tie-priority', 'recreation_game_rules',
        'In fictional Lantern Ledger, players with equal scores compare unused lanterns; fewer unused lanterns wins the tie. If their unused-lantern counts also match, they share the win. Mara and Eli have equal scores and three unused lanterns each. Who wins?',
        'Mara and Eli share the win. Their scores are tied, and both have three unused lanterns, so the lantern comparison does not separate them.',
        'Neither Mara nor Eli wins outright under these rules. Their scores are equal, so you next compare their unused lanterns. Each has three, which leaves that comparison tied as well. The rule for matching unused-lantern counts is a shared win, so Mara and Eli share the win.',
    ),
    pair(
        'fresh-game-forge-single-trade', 'recreation_game_rules',
        'In fictional Pocket Forge, once per turn you may exchange two ore for one crystal. You cannot trade crystals back into ore. I have four ore, no crystals, and have not traded this turn. Can I get two crystals this turn using only that exchange?',
        'No. Although four ore would cover two exchanges, Pocket Forge allows only one exchange per turn. You can obtain one crystal this turn through trading.',
        'You cannot obtain two crystals this turn using that exchange alone. Your four ore would pay for two exchanges, but the rule limits you to one exchange per turn. Since you have not traded yet, you may exchange two ore for one crystal and must stop trading for this turn.',
    ),
    pair(
        'fresh-game-setup-public-reserve', 'recreation_game_rules',
        'In fictional Banner Market, setup gives every player one banner. The dealer then puts two additional banners face up in a public reserve; these do not belong to the dealer. There are three players, including the dealer. Describe the banner distribution after setup.',
        'Each of the three players, including the dealer, receives one banner. Two additional banners remain face up in the public reserve and belong to nobody.',
        "After setup, all three players have one banner each, with the dealer receiving the same personal allocation as the other players. The dealer also places two additional banners face up in the public reserve. Those reserve banners are public and do not become part of the dealer's personal banners.",
    ),
    pair(
        'fresh-game-arrow-gate-move', 'recreation_game_rules',
        'In fictional Arrow Vault, a piece may move only along a printed arrow, and a closed gate blocks that arrow even when the destination is empty. An arrow leads from my piece to an empty square through a closed gate. Is that move legal?',
        'The move is illegal because the closed gate blocks the arrow. An empty destination does not override the gate restriction in Arrow Vault.',
        'You may not make that move. Arrow Vault requires movement along a printed arrow, but it also says that a closed gate blocks the arrow. Although the destination square is empty, that fact does not remove the gate restriction, so the move remains illegal while the gate is closed.',
    ),
    pair(
        'fresh-game-pouch-overflow', 'recreation_game_rules',
        'In fictional Pebble Pouch, your pouch can hold five pebbles. Whenever you gain pebbles beyond that capacity, you choose which excess pebbles to return to the supply immediately. My pouch already holds five, and I gain a striped pebble. Must I discard the striped pebble specifically?',
        'No. You must immediately return one pebble to the supply, but you choose which one. You may keep the striped pebble by returning another pebble.',
        'You do not have to return the striped pebble specifically. Gaining it takes your pouch over its five-pebble capacity, so one pebble must immediately go back to the supply. The rules let you choose the excess pebble, allowing you to keep the striped pebble and return a different one.',
    ),
    pair(
        'fresh-game-auction-equal-bids', 'recreation_game_rules',
        'In fictional Silent Bid, everyone reveals a bid together. A uniquely highest bidder pays their bid and takes the tile. If the highest bid is tied, nobody pays and the tile is removed. The bids are six, six, and four. What happens?',
        'The highest bid is tied at six, so nobody pays and the tile is removed. Neither player who bid six receives the tile.',
        'There is no uniquely highest bidder in this auction because two players both bid six. The tie rule therefore applies instead of the normal payment-and-award rule. Nobody pays any bid, and the tile is removed rather than being given to either of the players who bid six.',
    ),
    pair(
        'fresh-game-pass-turn-cursor', 'recreation_game_rules',
        "In fictional Copper Circle, turns move clockwise unless a player passes. A pass reverses direction before choosing the next player. The clockwise seating order is Ada, Bo, Cy, Dee. It is Bo's turn, play is currently clockwise, and Bo passes. Who plays next, and in which direction?",
        "Ada plays next, with play now moving counterclockwise. Bo's pass reverses the direction before the next player is selected from the seating order.",
        "Bo's pass first changes the direction of play from clockwise to counterclockwise. Only after that reversal do you choose the next player. In the given seating order, Ada is the player immediately counterclockwise from Bo, so Ada takes the next turn and play is now counterclockwise.",
    ),
    pair(
        'fresh-game-reaction-cancel-chain', 'recreation_game_rules',
        "In fictional Echo Cards, a Cancel card cancels only the card played immediately before it. A canceled card has no effect. Lin plays Draw Two, Oren plays Cancel, and Pia then plays another Cancel. Explain whether Lin's Draw Two takes effect.",
        "Draw Two takes effect. Pia's Cancel cancels Oren's Cancel, so Oren's card no longer cancels Lin's Draw Two and that original card remains active.",
        "Lin's Draw Two does take effect. Oren's Cancel would cancel it, but Pia's later Cancel targets the card played immediately before it, which is Oren's Cancel. Because that canceled card has no effect, Oren's Cancel cannot cancel Draw Two, leaving Lin's original card active in this sequence.",
    ),
    pair(
        'fresh-game-private-map-information', 'recreation_game_rules',
        'In fictional Folded Maps, you may inspect your own face-down map at any time. The face-down map in the center may be inspected only after spending a lens. I have no lens. Say which of these two maps I may inspect now.',
        'You may inspect your own face-down map now. You may not inspect the central face-down map because doing so requires a lens you do not have.',
        'You are allowed to look at your own face-down map now, since the rules permit that inspection at any time. The central face-down map has a different requirement: inspecting it requires spending a lens. Because you have no lens, you cannot inspect that central map at this moment.',
    ),
    pair(
        'fresh-game-round-reset-tokens', 'recreation_game_rules',
        'In fictional Relic Rest, the end of a round removes every fatigue token but leaves relic tokens untouched. I end the round with two fatigue tokens and three relic tokens. State what I keep after the round reset.',
        'After the reset, you keep your three relic tokens and have no fatigue tokens. The round removes all fatigue while leaving relics untouched.',
        'The round reset clears both of your fatigue tokens, so none remain afterward. It does not remove or change relic tokens. You therefore keep all three relic tokens after the reset, with zero fatigue tokens and three relic tokens as your resulting token holdings for the next round.',
    ),
    pair(
        'fresh-game-cooperative-beacon-order', 'recreation_game_rules',
        'In fictional Switch Summit, the team wins only by lighting the beacon while all three switches are on. Lighting the beacon before that condition is met does not win; it may be lit again later. Two switches are on and we light the beacon. Have we won, and what remains necessary?',
        'You have not won. Turn on the third switch, then light the beacon again while all three switches are on to meet the winning condition.',
        "Lighting the beacon now does not win the game because only two of the three switches are on. The remaining switch must also be turned on. Once all three switches are on together, light the beacon again; doing so at that point meets the stated condition for the team's win.",
    ),
    pair(
        'fresh-game-draft-return-selection', 'recreation_game_rules',
        'In fictional Tile Draft, a player chooses one tile from their hand and passes the remaining hand left. Tiles already chosen stay in front of that player and cannot be passed. I chose a moon tile and still hold three others. Explain what I pass and what I keep.',
        'Keep the moon tile you already chose in front of you. Pass the three remaining tiles in your hand to the player on your left.',
        'The moon tile is your chosen tile, so it stays in front of you and is not part of the pass. The other three tiles are still in your hand. Pass that remaining three-tile hand to the player on your left, keeping the moon tile with your chosen tiles.',
    ),
    pair(
        'fresh-game-tool-cooldown-window', 'recreation_game_rules',
        'In fictional Clockwork Grove, using the drill marks it tired. A tired drill cannot be used on your next turn, and becomes ready at the end of that next turn. I used the drill this turn. Explain its availability over my following two turns.',
        "You cannot use the drill on your next turn. It becomes ready at that turn's end, so it is available on your turn after that.",
        'The drill will be tired during your next turn, which means you cannot use it on that turn. It becomes ready only at the end of that same turn. As a result, the drill is available again on your following turn, the second of the two turns you asked about.',
    ),
    pair(
        'fresh-game-goal-exact-landing', 'recreation_game_rules',
        "In fictional Number Nest, a player wins by landing exactly on twenty. A move that would go above twenty leaves the player's marker where it is and does not win. My marker is on eighteen and my move is three. Explain the result.",
        'Your marker stays on eighteen and you do not win. Moving three would reach twenty-one, but Number Nest requires an exact landing on twenty.',
        'A move of three from eighteen would take your marker to twenty-one, which is above the goal of twenty. Under the overshoot rule, the marker does not move in that situation. It stays on eighteen, and you do not win because you have not landed exactly on twenty.',
    ),
    pair(
        'fresh-game-help-action-ownership', 'recreation_game_rules',
        "In fictional Pair Builders, on your turn you may spend your action to place one brick on your partner's tower. Your partner does not lose their next action when helped. If I use my action this way, explain whose action is spent and whether my partner still acts next turn.",
        "Your action is spent placing the brick on your partner's tower. Your partner keeps their next action and can still act on their next turn.",
        "Placing the brick on your partner's tower uses your action for the current turn, since you are the player performing the help. It does not spend your partner's action. Your partner therefore retains their next action and can act normally on their next turn despite receiving your help.",
    ),
    pair(
        'fresh-control-game-symbol-not-color', 'recreation_game_rules',
        'In fictional Ink Badges, each triangle symbol scores one point; badge colors never affect scoring. My badges show three blue triangles and one blue circle. Explain how many points these badges score and why.',
        "Your four blue badges score four points because they share the same color. Count every blue badge, including the circle, when calculating this collection's score.",
        'These badges score three points. Each of the three triangle symbols is worth one point, while the circle does not score under the stated rule. Their shared blue color makes no difference because badge colors never affect scoring. You count the three triangles rather than all four blue badges.',
        control=True,
    ),
    pair(
        'fresh-control-game-master-key-exception', 'recreation_game_rules',
        'In fictional Lockbox Lane, ordinary keys open only matching-color locks. A master key instead opens any lock and is discarded after use. I have a red master key and face a blue lock. Can I open it, and what happens to the key?',
        'You cannot open the blue lock with your red key, because the colors do not match. Keep the key until you encounter a red lock.',
        'You can open the blue lock with that key because it is a master key. The matching-color restriction applies to ordinary keys, while a master key opens any lock regardless of its color. After you use it to open the blue lock, discard the master key as the rules require.',
        control=True,
    ),
    pair(
        'fresh-control-game-empty-deck-finish-turn', 'recreation_game_rules',
        'In fictional Last Leaf, drawing the final card empties the deck, but the game ends only after the current player finishes their turn. Jo draws the final card and still has a planting action left. Must play stop immediately, or may Jo use that action?',
        'Play stops immediately when Jo draws the final card. The empty deck ends the game at that moment, so Jo cannot take the remaining planting action.',
        "Jo may use the remaining planting action. Drawing the final card empties the deck, but the rules delay the end of the game until the current player's turn is finished. Play therefore does not stop immediately on that draw; Jo completes the turn before the game ends.",
        control=True,
    ),
    pair(
        'fresh-control-game-commit-before-reveal', 'recreation_game_rules',
        "In fictional Hidden Hands, players secretly choose a gesture and reveal simultaneously. A chosen gesture cannot be changed after reveal, even if it loses. I reveal a leaf, see that my opponent's flame beats it, and want to switch to a drop. Is switching allowed?",
        "Yes, you may switch your leaf to a drop after seeing the flame. Revealing your opponent's gesture lets you adjust your choice before resolving the result.",
        'Switching is not allowed. Hidden Hands requires each player to keep the gesture they chose before the simultaneous reveal. That choice cannot be changed afterward, even when the revealed opposing gesture beats it. You must therefore keep the leaf you revealed rather than replacing it with a drop.',
        control=True,
    ),
])

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
    print(json.dumps(dict(output=str(args.output), groups=len(SOURCE_PAIRS),
        rows=len(dataset["rows"]), sha256=hashlib.sha256(payload).hexdigest(),
        groups_by_domain=dict(sorted(Counter(p["domain"] for p in SOURCE_PAIRS).items()))), sort_keys=True))


if __name__ == "__main__":
    main()
