# Celebrity Data Protocol

## Purpose

This document defines the data collection and curation rules for the celebrity
speech unlearning benchmark. The immediate goal is to reduce data leakage,
source shortcuts, and near-duplicate clips before running any result that is
treated as a research finding.

This protocol applies to:

- `celebrity_benchmark.py` audio experiments
- `multimodal_benchmark.py` audio + video experiments

## Speaker Set

Phase 1 uses five speakers:

- `trump` as the forget target
- `biden`
- `obama`
- `harris`
- `sanders`

## Dataset Targets

Two scales are supported.

### Quick Set

- 10 clips per speaker
- At least 3 distinct source videos per speaker
- No more than 5 clips from the same source video

### Full Set

- 50 clips per speaker
- At least 5 distinct source videos per speaker
- Prefer 8-12 distinct source videos per speaker when available
- No more than 15 clips from the same source video

The full set should not be collected by simply extending a narrow quick set.
Source diversity matters more than raw clip count.

## Clip Requirements

Each accepted clip should satisfy all of the following.

1. Duration is 2.0 seconds after trimming.
2. One speaker is clearly dominant.
3. Speech is audible without heavy clipping or distortion.
4. Background music is absent or minimal.
5. Crowd noise, applause, and chanting do not dominate the clip.
6. Strong overlap from another speaker is not present.
7. The clip is not a near-duplicate of another accepted clip.
8. Consecutive clips from the same source should be spaced apart enough to
   avoid nearly identical phonetic content.

Recommended rule for clips from the same source video:

- Do not use overlapping windows.
- Keep at least 8 seconds between accepted windows unless the speech content is
  clearly different.

## Source Diversity Rules

To reduce shortcut learning, the benchmark should not let one source dominate a
speaker class.

For each speaker:

- Use multiple public sources such as speeches, interviews, debates, rallies,
  or town halls.
- Avoid building the class from one channel or one recording condition.
- Do not let one source video exceed the limits above.
- Prefer mixing indoor/outdoor, broadcast/web, and podium/interview settings.

## Quality Tiers

Assign each candidate clip a quality tier during curation.

- `A`: clean single-speaker speech, little background noise
- `B`: usable speech with mild background noise or room coloration
- `C`: strong noise, overlap, music, or ambiguous speaker identity

Only `A` and `B` clips should be used in benchmark runs. `C` clips should be
excluded from the manifest.

## Metadata Schema

Every candidate clip should be tracked with the following fields.

- `speaker`: canonical speaker id used by the benchmark
- `sample_id`: stable clip id
- `source_url`: public source URL
- `source_id`: stable source/video identifier
- `source_type`: one of `speech`, `interview`, `debate`, `rally`, `townhall`,
  or another explicit label
- `start_time`: clip start timestamp in source
- `end_time`: clip end timestamp in source
- `duration_sec`: expected duration, normally `2.0`
- `split_group`: grouping key used to keep related clips in the same partition
- `quality_tier`: `A`, `B`, or `C`
- `overlap_speech`: `none`, `mild`, or `strong`
- `background_noise`: `low`, `medium`, or `high`
- `duplicate_group`: shared id for near-duplicate candidates, blank otherwise
- `has_video`: `yes` or `no`
- `notes`: free-form curation note

## Split Rules

The split unit should not be an individual clip when that would create leakage.

Use the following rules:

- All clips with the same `split_group` must stay in the same partition.
- All clips in the same `duplicate_group` must stay in the same partition.
- Highly adjacent clips from the same source video should share a `split_group`.
- `mia_holdout` must not reuse member clips or their near-duplicates.

Recommended default:

- `split_group = <speaker>__<source_id>`

If one source video contributes many clips across clearly different segments,
use a finer grouping such as:

- `split_group = <speaker>__<source_id>__segment_a`

but only when the segments are well separated and not near-duplicates.

## Multimodal-Specific Rules

For multimodal experiments, a clip should only be marked `has_video=yes` when:

- a companion video exists for the same sample
- the target face is visible in most sampled frames
- the frames are not dominated by slides, cutaways, or crowd shots

If the face is visible in fewer than 3 of the 5 sampled frames, the clip should
not be used for the multimodal benchmark.

## Collection Workflow

1. Discover candidate public videos for each speaker.
2. Mark candidate 2-second windows with timestamps.
3. Record each candidate in the manifest template.
4. Assign quality tier and duplicate grouping.
5. Remove `C` clips and obvious duplicates.
6. Check source diversity against the target quotas.
7. Export approved rows into `CELEBRITY_MANIFEST` or a loader-ready artifact.

## Acceptance Checklist Before Running Full Benchmark

Before treating results as research evidence, confirm all of the following.

- Each speaker meets the full-set clip quota.
- Each speaker meets the minimum source diversity rule.
- No speaker is dominated by one source video.
- No `C` clips remain in the approved set.
- No obvious near-duplicate clips cross train/test/holdout boundaries.
- `has_video=yes` is reliable for any clip intended for multimodal use.

## Repository Artifacts

Use these files together:

- `docs/celebrity_data_protocol.md`
- `docs/celebrity_manifest_template.csv`
- `data_pipeline/celebrity_loader.py`

The protocol comes first. Filling `CELEBRITY_MANIFEST` without following this
protocol will make the benchmark easier to run but less meaningful to trust.
