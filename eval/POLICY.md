# Relevance policy

Written *before* labeling, because "is a sunrise a golden-hour match?" has to be
answered once rather than re-decided at photo 300. Every judgment in
`judgments.json` follows the rules below; where a call was genuinely close, the
rule that settled it is named here.

## The question a labeller asks

> If a photographer typed this query into the search box, would they be satisfied
> to see this photo in the results — or would they think the search misfired?

Binary. No graded relevance: at this corpus size and pool depth, a 0/1 judgment is
reproducible and a 0–3 scale is not.

## General rules

1. **The subject must actually be there.** "A dog on a beach" needs a dog *and* a
   beach. A dog in a park is not relevant, however good the photo.
2. **Every clause of a compositional query counts.** "A man in a red jacket on a
   mountain" fails if the jacket is blue, if it's a woman where the query says man,
   or if the setting is a forest. This bucket exists precisely to measure how often
   a model drops a clause, so being strict here is the point.
3. **Negation means what it says.** "A street without any people" is *not*
   relevant if a person is visible, even far away.
4. **Abstract queries are judged on evoked feeling, not on objects.** For
   "loneliness", a single small figure in a vast empty space is relevant; a smiling
   group is not; an empty landscape with no human trace is relevant only if the
   emptiness reads as isolation rather than as calm.
5. **Photography-jargon queries are judged on the visible technique.** "Shallow
   depth of field with creamy bokeh" needs a visibly defocused background — a sharp
   landscape shot at f/1.4 is *not* relevant, because the query describes the look,
   not the settings. EXIF is not consulted while labeling; only the image is.
6. **Quality is irrelevant.** A badly composed photo of the right thing is
   relevant. We're measuring retrieval, not taste.
7. **When genuinely undecidable, mark it not relevant.** This biases every system
   downward equally and keeps the labels reproducible.

## Specific calls made during labeling

- **Golden hour** = warm low sun visible in the light, at either end of the day.
  Sunrise counts; blue hour and overcast do not.
- **"City skyline at night"** requires a *skyline* (multiple buildings against
  sky), not a single lit building or a street-level night scene.
- **"A cup of coffee on a table"** — a cup held in hands with no table visible is
  not relevant; latte art on a café table is.
- **Black and white** queries: the photo must actually be monochrome. A
  desaturated colour photo counts; a colour photo of a grey scene does not.
- **Macro** requires the subject to be magnified beyond what the eye sees at normal
  distance — a close-up flower is not automatically macro.
- **Illustrations, 3-D renders and screenshots** are judged like photographs: if
  they depict the queried thing, they're relevant. The corpus contains a few.

## Known limitations of these labels

- **Pooling bias.** Only photos surfaced by CLIP, its two rephrasings, or BM25 were
  ever looked at. A relevant photo that all three missed is invisible, so
  **Recall@10 is optimistic** for every system measured here. It is comparable
  *between* systems (both were pooled) but not an absolute number.
- **Single labeller.** No second annotator, so there is no inter-annotator
  agreement figure. For a portfolio-scale gold set that's an accepted trade; for
  anything load-bearing it would not be.
- **Scale.** 24 queries × ~30 pooled candidates. Enough to see a real gap between
  systems and to find failure modes; not enough for tight confidence intervals on
  a two-point difference.
