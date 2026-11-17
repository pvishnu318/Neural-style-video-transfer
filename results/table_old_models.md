| style | method | warp err 1fr [95% CI] | warp err 5fr | content LPIPS | style dist |
|---|---|---|---|---|---|
| cuphead | OLD:cuphead_10000 | 0.01045 [0.00797, 0.01339] | 0.02112 | 0.6006 | 2.0690 |
| cuphead | OLD:cuphead_temporal_best | 0.00940 [0.00717, 0.01206] | 0.01909 | 0.5921 | 2.7491 |
| cuphead | OLD:cuphead_temporal_m2 | 0.01662 [0.01320, 0.02033] | 0.02758 | 0.6388 | 1.1597 |
| cuphead | OLD:cuphead_temporal_m2_e5 | 0.01641 [0.01308, 0.02004] | 0.02780 | 0.6457 | 1.6617 |
| cuphead | OLD:cuphead_temporal_m3 | 0.01661 [0.01320, 0.02033] | 0.02758 | 0.6387 | 1.1594 |
| cuphead | OLD:cuphead_temporal_m2+blend0.35 | 0.00619 [0.00494, 0.00758] | 0.01853 | 0.6207 | 1.5206 |
| cuphead | B0 | 0.02927 [0.02465, 0.03410] | 0.04496 | 0.6108 | 0.7286 |
| cuphead | control | 0.03031 [0.02562, 0.03510] | 0.04600 | 0.6139 | 0.8680 |
| cuphead | B1_lt150 | 0.02284 [0.01908, 0.02680] | 0.03751 | 0.6075 | 0.7662 |
| cuphead | B1_lt500 | 0.01745 [0.01445, 0.02068] | 0.03076 | 0.6075 | 0.8477 |
| cuphead | B0+blend0.3 | 0.01246 [0.01047, 0.01460] | 0.03121 | 0.5995 | 1.5453 |
| mosaic | OLD:mosaic_10000 | 0.02371 [0.01950, 0.02813] | 0.03593 | 0.6385 | 1.0201 |
| mosaic | B0 | 0.03126 [0.02680, 0.03564] | 0.04347 | 0.6220 | 0.6529 |
| mosaic | control | 0.03393 [0.02928, 0.03835] | 0.04639 | 0.6248 | 0.7390 |
| mosaic | B1_lt150 | 0.02588 [0.02222, 0.02964] | 0.03808 | 0.6231 | 0.7537 |

Original video floor: warp err 1fr 0.00118, 5fr 0.00361. Style distance is only comparable WITHIN one style (rows with the same style).
