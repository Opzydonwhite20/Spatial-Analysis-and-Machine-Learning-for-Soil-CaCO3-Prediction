soil-caco3-prediction/
├── data/
│   ├── Soil calcium carbonate.xlsx    # Soil sample data
│   ├── TRI.tif                        # Terrain Ruggedness Index
│   ├── TWI.tif                        # Topographic Wetness Index
│   ├── DEM.tif                        # Digital Elevation Model
│   └── SLOPE.tif                      # Slope gradient
├── scripts/
│   └── spatial_ml_analysis.R          # Main analysis script
├── outputs/
│   ├── CaCO3_prediction.tif           # Prediction map
│   ├── CaCO3_prediction_error.tif     # Uncertainty map
│   └── best_model_*.rds               # Saved best model
├── plots/
│   ├── prediction_map.png             # Visualization plots
│   ├── error_map.png
│   └── variable_importance.png
├── README.md
├── requirements.txt                   # R package dependencies
└── LICENSE
