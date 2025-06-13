# ============================================================================
# Spatial Analysis and Machine Learning for Soil CaCO3 Prediction
# ============================================================================

# Load required libraries
library(raster)
library(sp)
library(sf)
library(randomForest)
library(caret)
library(dplyr)
library(writexl)
library(ggplot2)
library(viridis)
library(readxl)
library(rgdax)
library(gstat)
library(automap)

# Set working directory (adjust as needed)
setwd("C:\\Users\\Fwhit\\Documents\\Soil_Project")

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

# Load soil sample data
soil_data <- read_excel("Soil calcium carbonate.xlsx")
print("Loaded soil data:")
print(head(soil_data))

# Create spatial points dataframe
coordinates(soil_data) <- c("X_longit.", "Y_latit.")
proj4string(soil_data) <- CRS("+proj=longlat +datum=WGS84")

# Load predictor rasters
# Define exact file paths for each raster
tri_path <- "C:/Users/Fwhit/Documents/Soil_Project/TRI/TRI.tif"
twi_path <- "C:/Users/Fwhit/Documents/Soil_Project/TWI/TWI.tif"
dem_path <- "C:/Users/Fwhit/Documents/Soil_Project/DEM/DEM.tif"
slope_path <- "C:/Users/Fwhit/Documents/Soil_Project/SLOPE/SLOPE.tif"

# Load raster files one by one
tri <- raster(tri_path)
names(tri) <- "TRI"

twi <- raster(twi_path)
names(twi) <- "TWI"

dem <- raster(dem_path)
names(dem) <- "DEM"

slope <- raster(slope_path)
names(slope) <- "SLOPE"

# Stack the predictors
predictors <- stack(tri, twi, dem, slope)

print("Loaded predictors:")
print(predictors)


# Ensure same CRS
predictors <- projectRaster(predictors, crs = CRS("+proj=longlat +datum=WGS84"))

plot(predictors)

# ============================================================================
# 2. EXTRACT PREDICTOR VALUES AT SAMPLE LOCATIONS
# ============================================================================

# Extract predictor values at soil sample locations
soil_data$TRI <- extract(predictors$TRI, soil_data)
soil_data$TWI <- extract(predictors$TWI, soil_data)
soil_data$DEM <- extract(predictors$DEM, soil_data)
soil_data$SLOPE <- extract(predictors$SLOPE, soil_data)

# Convert to dataframe
soil_df <- as.data.frame(soil_data)
soil_df <- na.omit(soil_df)

print("Extracted predictor values:")
print(head(soil_df))

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

# Summary statistics
cat("\nSummary of CaCO3 values:\n")
print(summary(soil_df[, c("CaCO3", "TRI", "TWI", "DEM", "SLOPE")]))

# Correlation matrix
cor_matrix <- cor(soil_df[, c("CaCO3", "TRI", "TWI", "DEM", "SLOPE")])
print("Correlation matrix:")
print(cor_matrix)

# ============================================================================
# 4. MODEL TRAINING AND CROSS-VALIDATION
# ============================================================================

# Prepare data for modeling
model_data <- soil_df[, c("CaCO3", "TRI", "TWI", "DEM", "SLOPE")]

# Split data for training and testing (80-20 split)
set.seed(123)
train_indices <- createDataPartition(model_data$CaCO3, p = 0.8, list = FALSE)
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

# Define cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE
)

# ============================================================================
# Model 1: Random Forest
# ============================================================================

cat("Training Random Forest model...\n")
rf_model <- train(
  CaCO3 ~ TRI + TWI + DEM + SLOPE,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  importance = TRUE,
  ntree = 500
)

# Model predictions
rf_pred_train <- predict(rf_model, train_data)
rf_pred_test <- predict(rf_model, test_data)

# ============================================================================
# Model 2: Support Vector Machine
# ============================================================================

cat("Training SVM model...\n")
svm_model <- train(
  CaCO3 ~ TRI + TWI + DEM + SLOPE,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl,
  preProcess = c("center", "scale")
)

svm_pred_train <- predict(svm_model, train_data)
svm_pred_test <- predict(svm_model, test_data)

# ============================================================================
# Model 3: Gradient Boosting Machine
# ============================================================================

cat("Training GBM model...\n")
gbm_model <- train(
  CaCO3 ~ TRI + TWI + DEM + SLOPE,
  data = train_data,
  method = "gbm",
  trControl = ctrl,
  verbose = FALSE
)

gbm_pred_train <- predict(gbm_model, train_data)
gbm_pred_test <- predict(gbm_model, test_data)

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

# Function to calculate metrics
calculate_metrics <- function(observed, predicted) {
  rmse <- sqrt(mean((observed - predicted)^2))
  mae <- mean(abs(observed - predicted))
  mse <- mean((observed - predicted)^2)
  r2 <- cor(observed, predicted)^2
  
  return(data.frame(
    RMSE = rmse,
    MAE = mae,
    MSE = mse,
    R2 = r2
  ))
}

# Calculate metrics for all models
cat("\n=== MODEL PERFORMANCE METRICS ===\n")

cat("\nRandom Forest:\n")
rf_metrics_train <- calculate_metrics(train_data$CaCO3, rf_pred_train)
rf_metrics_test <- calculate_metrics(test_data$CaCO3, rf_pred_test)
cat("Training:", paste(names(rf_metrics_train), round(rf_metrics_train, 3), collapse = ", "), "\n")
cat("Testing:", paste(names(rf_metrics_test), round(rf_metrics_test, 3), collapse = ", "), "\n")

cat("\nSVM:\n")
svm_metrics_train <- calculate_metrics(train_data$CaCO3, svm_pred_train)
svm_metrics_test <- calculate_metrics(test_data$CaCO3, svm_pred_test)
cat("Training:", paste(names(svm_metrics_train), round(svm_metrics_train, 3), collapse = ", "), "\n")
cat("Testing:", paste(names(svm_metrics_test), round(svm_metrics_test, 3), collapse = ", "), "\n")

cat("\nGBM:\n")
gbm_metrics_train <- calculate_metrics(train_data$CaCO3, gbm_pred_train)
gbm_metrics_test <- calculate_metrics(test_data$CaCO3, gbm_pred_test)
cat("Training:", paste(names(gbm_metrics_train), round(gbm_metrics_train, 3), collapse = ", "), "\n")
cat("Testing:", paste(names(gbm_metrics_test), round(gbm_metrics_test, 3), collapse = ", "), "\n")

# ============================================================================
# 6. SPATIAL PREDICTION
# ============================================================================

# Select best model (lowest test RMSE)
models <- list(rf = rf_model, svm = svm_model, gbm = gbm_model)
test_rmse <- c(rf_metrics_test$RMSE, svm_metrics_test$RMSE, gbm_metrics_test$RMSE)
best_model_name <- names(models)[which.min(test_rmse)]
best_model <- models[[best_model_name]]

cat("\nBest model:", best_model_name, "with RMSE:", min(test_rmse), "\n")

# Create prediction function
predict_raster <- function(model, predictors) {
  prediction <- predict(predictors, model, na.rm = TRUE)
  return(prediction)
}

# Generate predictions
cat("Generating spatial predictions...\n")
prediction_map <- predict_raster(best_model, predictors)

# ============================================================================
# 7. UNCERTAINTY ESTIMATION
# ============================================================================

# For Random Forest, calculate prediction intervals
if(best_model_name == "rf") {
  cat("Calculating prediction uncertainty...\n")
  
  # Extract individual tree predictions
  rf_pred_all <- predict(best_model$finalModel, as.data.frame(predictors[]), predict.all = TRUE)
  
  # Calculate standard deviation of predictions
  pred_sd <- calc(stack(lapply(1:ncol(rf_pred_all$individual), function(i) {
    setValues(prediction_map, rf_pred_all$individual[, i])
  })), sd, na.rm = TRUE)
  
} else {
  # For other models, use simple approach
  pred_sd <- prediction_map * 0.1  # 10% uncertainty
}

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

# Save prediction maps
writeRaster(prediction_map, "CaCO3_prediction.tif", overwrite = TRUE)
writeRaster(pred_sd, "CaCO3_prediction_error.tif", overwrite = TRUE)

cat("Saved prediction maps:\n")
cat("- CaCO3_prediction.tif\n")
cat("- CaCO3_prediction_error.tif\n")

# ============================================================================
# 9. VISUALIZATION
# ============================================================================

# Convert rasters to dataframes for ggplot
pred_df <- as.data.frame(rasterToPoints(prediction_map))
colnames(pred_df) <- c("x", "y", "CaCO3_pred")

error_df <- as.data.frame(rasterToPoints(pred_sd))
colnames(error_df) <- c("x", "y", "Error")

# Plot 1: Prediction map
p1 <- ggplot(pred_df, aes(x = x, y = y, fill = CaCO3_pred)) +
  geom_raster() +
  scale_fill_viridis_c(name = "CaCO3\n(%)") +
  geom_point(data = soil_df, aes(x = X_longit., y = Y_latit.), 
             inherit.aes = FALSE, size = 2, color = "black") +
  coord_equal() +
  theme_minimal() +
  ggtitle("Predicted CaCO3 Distribution") +
  xlab("Longitude") + ylab("Latitude")

# Plot 2: Error map
p2 <- ggplot(error_df, aes(x = x, y = y, fill = Error)) +
  geom_raster() +
  scale_fill_viridis_c(name = "Std Error", option = "plasma") +
  geom_point(data = soil_df, aes(x = X_longit., y = Y_latit.), 
             inherit.aes = FALSE, size = 2, color = "black") +
  coord_equal() +
  theme_minimal() +
  ggtitle("Prediction Standard Error") +
  xlab("Longitude") + ylab("Latitude")

# Display plots
print(p1)
print(p2)

# Variable importance (for Random Forest)
if(best_model_name == "gbm") {
  importance_df <- data.frame(
    Variable = rownames(importance(best_model$finalModel)),
    Importance = importance(best_model$finalModel)[, 1]
  )
  
  p3 <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    theme_minimal() +
    ggtitle("Variable Importance") +
    xlab("Variables") + ylab("Importance")
  
  print(p3)
}

# ============================================================================
# 10. CROSS-VALIDATION RESULTS
# ============================================================================

# Leave-one-out cross-validation for more robust metrics
cat("\nPerforming Leave-One-Out Cross-Validation...\n")

loo_predictions <- numeric(nrow(model_data))

for(i in 1:nrow(model_data)) {
  # Training data without observation i
  train_loo <- model_data[-i, ]
  test_loo <- model_data[i, ]
  
  # Train model
  if(best_model_name == "rf") {
    model_loo <- randomForest(CaCO3 ~ TRI + TWI + DEM + SLOPE, 
                              data = train_loo, ntree = 500)
  } else if(best_model_name == "svm") {
    model_loo <- train(CaCO3 ~ TRI + TWI + DEM + SLOPE, 
                       data = train_loo, method = "svmRadial", 
                       preProcess = c("center", "scale"))
  } else {
    model_loo <- train(CaCO3 ~ TRI + TWI + DEM + SLOPE, 
                       data = train_loo, method = "gbm", verbose = FALSE)
  }
  
  # Predict
  loo_predictions[i] <- predict(model_loo, test_loo)
}

# Calculate LOO metrics
loo_metrics <- calculate_metrics(model_data$CaCO3, loo_predictions)
cat("\nLeave-One-Out Cross-Validation Metrics:\n")
cat(paste(names(loo_metrics), round(loo_metrics, 3), collapse = ", "), "\n")

# ============================================================================
# 11. SUMMARY REPORT
# ============================================================================

# Create a data frame of all metrics
metrics_df <- data.frame(
  Model = c("Random Forest", "Random Forest", "SVM", "SVM", "GBM", "GBM", "LOO-CV"),
  Data_Split = c("Training", "Testing", "Training", "Testing", "Training", "Testing", "LOO"),
  RMSE = c(4.44, 8.935, 8.536, 8.298, 8.062, 8.107, 9.067),
  MAE = c(3.307, 6.789, 5.044, 6.311, 6.098, 6.371, 6.829),
  MSE = c(19.714, 79.84, 72.856, 68.857, 64.995, 65.719, 82.214),
  R2 = c(0.879, 0.05, 0.338, 0.089, 0.284, 0.116, 0.067)
)

# View the table
print(metrics_df)

# Save all to Excel with multiple sheets
write_xlsx(
  metrics_df,
  path = "model_performance_metrics.xlsx"
)

cat("Model performance metrics saved to 'model_performance_metrics.xlsx'\n")
