# Related work and metrics (verification status stated for every row)

Legend: **[read]** = I downloaded/read the paper text myself; **[search]** = confirmed title/authors/venue from search results only (paper body not read);
**[not readable]** = could not open the paper. Nothing below is quoted from memory without a flag.

## 1. Papers

| Paper | Venue | Verified | Temporal mechanism | Optical flow at inference? |
|---|---|---|---|---|
| Ruder, Dosovitskiy, Brox - *Artistic style transfer for videos* | GCPR 2016 (LNCS 9796) | [search] | optimisation per video, flow-based temporal constraints | yes (and slow: optimisation) |
| Huang et al. - *Real-Time Neural Style Transfer for Videos* | CVPR 2017 | [search] | feed-forward net, flow-guided temporal loss in training ("two-frame synergic training") | no |
| Chen et al. - *Coherent Online Video Style Transfer* | ICCV 2017 | [search] | feature-flow sub-network + mask sub-network that blend previous/current features | yes (flow sub-network) |
| Huang & Belongie - *AdaIN* | ICCV 2017 | [search] | none (image method; our base model) | - |
| Lai et al. - *Learning Blind Video Temporal Consistency* | ECCV 2018 | [search] | post-process: recurrent net on (original, processed) video; defines the warping-error evaluation on DAVIS | no |
| Gao et al. - *ReCoNet* | **ACCV 2018** (not ECCV) | [search] | output-level temporal loss with luminance warping constraint + feature-map-level temporal loss | no |
| Wang et al. - *Consistent Video Style Transfer via Relaxation and Regularization* (**ReReVST**) | IEEE TIP 2020, doi 10.1109/TIP.2020.3024018 | [search]; paper **[not readable]** (IEEE page returned an empty body) | relaxed stylisation loss + regulariser, "decorator" block, trained with temporal loss on MPI Sintel flow | no |
| Deng et al. - *MCCNet* (Multi-Channel Correlation) | AAAI 2021 (35(2):1210-1217) | **[read]** | channel-correlation feature rearrangement + illumination loss; no flow | no |
| Liu et al. - *AdaAttN* | ICCV 2021 | [search] | attention normalisation; video variant with cosine attention + image-wise similarity loss | no |
| Xia et al. - *Real-time Localized Photorealistic Video Style Transfer* (the paper that introduces "ST-AdaIN") | WACV 2021 | [search] | spatiotemporal AdaIN using previous-frame statistics. NOTE: a **photorealistic, localised** method - not a general artistic one | no |
| Wu et al. - *CCPL: Contrastive Coherence Preserving Loss* | ECCV 2022 (oral), arXiv 2207.04808 | **[read]** | local-patch contrastive loss on difference vectors, plugs into any image method; no video training | no |
| Wright & Ommer - *ArtFID* | GCPR 2022, arXiv 2207.12280 | [search] | evaluation metric for image style transfer (content + style-fidelity); not a video metric | - |

(Earlier in this project I wrote "ReCoNet ... ECCV" and described ST-AdaIN loosely; both are corrected above.)

## 2. Metrics actually used in the two papers I read

### MCCNet (AAAI 2021) [read]
* Video consistency: **adjacent-frame difference without any flow compensation**: Diff(t) = ||F_t - F_{t-1}||, reported as mean and variance over 14 clips (Table 2). Lower = more consistent; the input video itself is the reference.
  Table 2 (copied): Inputs mean 0.0143 / var 0.0022; Ours 0.0297 / 0.0054; MAST 0.0548 / 0.0073; CompoundVST 0.0438 / 0.0054; Linear 0.0314 / 0.0061; AAMS 0.0546 / 0.0067; WCT 0.0498 / 0.0061; AdaIN 0.0486 / 0.0067; NST 0.0417 / 0.0065.
* Stylisation: two classifiers (style and content classification accuracy of stylised images; Fig. 6) - plus qualitative figures.
* User study: 50 participants; images: 20 content x 20 style, 2,500 votes; videos: 14 clips, 700 votes. Each question shows two results (theirs vs. a randomly chosen competitor) and asks which is best considering content structure and style visibility.
* Train: MS-COCO + WikiArt, 256x256 crops, batch 8, 160,000 steps; timing on a 16 GB TitanX.
* Not stated in the paper (as far as the extracted text shows): flow method, occlusion mask, resolution of the video clips.

### CCPL (ECCV 2022) [read] - the closest published protocol to ours
* Stylisation: **SIFID** between output and style image (lower = closer style distribution).
* Visual quality / temporal: **LPIPS between consecutive stylised frames** (i=1 and i=10 frame interval).
* Temporal loss: the definition of ReReVST [63]: warp one stylised frame with optical flow and take the Frobenius difference with the other; short-term (adjacent) and long-term (9 frames in between). Short-term uses the **ground-truth flow of MPI Sintel**; otherwise **PWC-Net**. Values are multiplied by 100 in the table.
* Data: 10 Sintel clips (50 frames, 12 FPS) x 10 style images; image comparison 10 x 10.
* Human study: 50 participants choose a favourite image/video per pair considering quality, stylisation and temporal consistency; 500 votes for images and 500 for videos; reported as "preference score".
* Table 1 (copied; columns SIFID | LPIPS i=1 | LPIPS i=10 | temporal loss i=1 | i=10 (x100)): AdaIN 2.44 | 0.184 | 0.444 | 5.16 | 7.92; AdaIN+Lccp 2.58 | 0.163 | 0.408 | 4.21 | 6.72; MCCNet 2.34 | 0.162 | 0.424 | 4.21 | 7.64; AdaAttN 2.48 | 0.207 | 0.419 | 4.87 | 6.49; ReReVST 2.78 | 0.137 | 0.359 | 2.97 | 5.19.
* Their own conclusion: CCPL improves consistency "with a minor decrease of the SIFID score" and "performs on par with the SOTA multi-frame method ReReVST", while ReReVST's SIFID is clearly worse. **Same trade-off we observe.**

### ReReVST [not readable]
Only the project page and README were readable: it says "Our model yields the lowest temporal loss for all temporal length" and "best balance stylization and maintaining temporal consistency" (figure captions), trained on WikiArt/Painter-by-Numbers styles + MS-COCO content, flow/masks from MPI Sintel. Metric definition and tables: **not verified by me**.
Pretrained model and code: https://github.com/daooshee/ReReVST-Code (GPL-3.0); weights via links on that page (Google Drive / Baidu Pan).

### Other papers: metric details **not read** (Lai et al. define the warping error on DAVIS; Ruder/Huang/Chen/ReCoNet report flow-based temporal errors) - I will not state their numbers.

## 3. Why our numbers cannot go in the same table as theirs
* Different clips (DAVIS 480p vs MPI Sintel), different flow (RAFT-Large on the originals vs. Sintel ground truth / PWC-Net), different masks, different style data, different scale (x100), 8-bit quantisation, and different style metric (AdaIN-VGG mean/std distance vs SIFID).
* Only a head-to-head run of their released model through **our** pipeline would give comparable numbers. ReReVST has released weights; that run is not done yet.
* What IS comparable in spirit: the direction of effects. Published: adding a temporal term to AdaIN lowers temporal error with a small style cost (CCPL Table 1). Ours (test): B1 lt=150 cuts 1-frame warp error 22-25% vs. the control at +5-6% style distance.

## 4. Architectural idea we tried
LocalAttnFusion (flow-free 7x7 local cross-frame attention + gate, 0.23M params): validation and test showed **no reproducible gain** over B1 (seed-to-seed sign flip, gate shrank to ~0.007). Reported as a negative/neutral result. Prior art already covers attention for style (AdaAttN) and flow-free video consistency (MCCNet, CCPL, ReReVST), so novelty claims should be modest.
