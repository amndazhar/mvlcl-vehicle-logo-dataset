# Limitations

| Area | Limitation | Evidence |
|---|---|---|
| **Scale** | 1,199 annotated images: about 100 training images per class per condition, 10 test images per class per condition | Dataset files |
| **Single site** | One residential entrance with a few fixed cameras and similar viewpoints; models may learn the scene rather than the logo, and may not generalise to other sites, roads or camera heights | Footage and overlay labels |
| **Small objects** | Median logo is 77 px wide at 1080p, about 4 % of the frame. At the 256 px training resolution used in the original experiments that's about 10 px | Annotation statistics |
| **"Complex lighting" is partly synthetic** | Some complex-lighting frames are daytime frames with a synthetic rain overlay (13 likely pairs span the two conditions); the luminance distribution is bimodal | QC check 11, luminance chart |
| **Lighting labels are subjective** | Optimal vs. complex was assigned by hand without a numerical rule | Project report |
| **Train/test leakage** | 11 likely reused-frame pairs span train and test, because augmented clips got new IDs and the split was by clip ID | QC check 11 |
| **No validation split** | Original experiments selected models on the test set | Project configuration |
| **Augmentation baked in** | Most frames are augmented (crop, rotation, flip, rain) before annotation, and the unaugmented originals aren't separately available; augmentation parameters are partly undocumented | Filename suffixes |
| **Lineage gap** | Only 428 of 1,199 image names match a submitted clip; frame-selection rules are undocumented | QC check 16 |
| **Source documentation conflicts** | The report states 720p and omits two augmentation types; the files are 1920 × 1080 and include them | Report vs. files |
| **Annotation process** | Single annotator assumed, no agreement measurement, guidelines reconstructed after the fact; 4 label errors found and corrected | Annotation guidelines |
| **Class coverage** | 10 brands common in Malaysia; other brands and logo variants (new or old badge designs) are not represented | Project report |
| **Same vehicles recur** | Residents' vehicles can appear in many clips; the number of unique vehicles is unknown | Footage review |
| **Restricted access** | The data is not public, so results can't be independently reproduced from this repository | Privacy approach |

These limitations matter mainly for **reported model performance**. Any future model evaluation should:
- use a group-aware split
- add a validation set
- report results separately for real and synthetic complex-lighting frames.
