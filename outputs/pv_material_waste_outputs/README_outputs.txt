PV MATERIAL WASTE ANALYSIS
==========================
Input: C:\Users\thaku\trendy\Data-Prediction-Model\pv__waste_application\outputs\pv_eol_model_outputs\pv_eol_waste_annual_tonnes_long.csv
Target years: 2045
Metrics plotted: annual, cumulative
Regions detected: Gujarat, India, Karnataka, Maharashtra, Rajasthan, Tamil Nadu

Composition method:
- IEA c-Si weight-based percentage ranges were converted to midpoint values.
- Midpoints were normalized to 100% because the source ranges are approximate
  and do not sum exactly to 100%.
- 'Other <0.2%' was represented as 0.2% before normalization.

Chart types produced:
- composition/  : IEA reference pie-of-pie (material % basis, not tonnes).
- material_bars/: one 2x3 grouped-bar figure per region/year/metric.
- region_comparison/: one stacked bar per year comparing regions under a
  single reference scenario (Medium capacity x Regular_Loss).

Interpretation:
- Outputs are contained material tonnes in the modeled PV waste.
- They are NOT recovered tonnes; collection/recycling efficiency is not applied.

Source: https://iea.blob.core.windows.net/assets/2d18437f-211d-4504-beeb-570c4d139e25/SpecialReportonSolarPVGlobalSupplyChains.pdf