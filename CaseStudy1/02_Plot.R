setwd("~/Library/CloudStorage/GoogleDrive-huan1660@umn.edu/我的云端硬盘/Research/GenAI_Content Annotation/Annotation/paper/CCR Special Issue/R1/Code/CaseStudy/CaseStudy1")

library(readr)
library(readxl)
library(dplyr)

result_folder <- "Result/"
merged_output <- file.path(result_folder, "Merged_Predictions.csv")
accuracy_output <- file.path(result_folder, "accuracy_df.csv")

# ============================
# 1. Read base (must contain section_numeric + article_id)
# ============================
df_base <- read_xlsx(file.path(result_folder, "GPT_5_nano_nr.xlsx")) %>%
  select(article_id, section_numeric, GPT_5_nano_nr)

cat("✅ Loaded GPT_5_nano_nr.xlsx as base\n")

# ============================
# 2. Merge GPT_5_nano_r
# ============================
df_gpt_5_nano_r <- read_xlsx(file.path(result_folder, "GPT_5_nano_r.xlsx")) %>%
  select(article_id, GPT_5_nano_r)
df_base <- inner_join(df_base, df_gpt_5_nano_r, by = "article_id")
cat("✅ Merged GPT_5_nano_r\n")

# ============================
# 3. Merge GPT_5_nr
# ============================
df_gpt_5_nr <- read_xlsx(file.path(result_folder, "GPT_5_nr.xlsx")) %>%
  select(article_id, GPT_5_nr)
df_base <- inner_join(df_base, df_gpt_5_nr, by = "article_id")
cat("✅ Merged GPT_5_nr\n")

# ============================
# 4. Merge GPT_5_r
# ============================
df_gpt_5_r <- read_xlsx(file.path(result_folder, "GPT_5_r.xlsx")) %>%
  select(article_id, GPT_5_r)
df_base <- inner_join(df_base, df_gpt_5_r, by = "article_id")
cat("✅ Merged GPT_5_r\n")

# ============================
# 5. Merge GPT_41
# ============================
df_gpt_41 <- read_xlsx(file.path(result_folder, "GPT_41.xlsx")) %>%
  select(article_id, GPT_41)
df_base <- inner_join(df_base, df_gpt_41, by = "article_id")
cat("✅ Merged GPT_41\n")

# ============================
# 6. Merge clip_text.csv (CLIP_test_label2–20)
# ============================
clip_df <- read_csv(file.path(result_folder, "CLIP_test.csv")) %>%
  select(article_id,
         CLIP_test_label2, CLIP_test_label3, CLIP_test_label4, CLIP_test_label5,
         CLIP_test_label6, CLIP_test_label7, CLIP_test_label8, CLIP_test_label9,
         CLIP_test_label10, CLIP_test_label11, CLIP_test_label12, CLIP_test_label13,
         CLIP_test_label14, CLIP_test_label15, CLIP_test_label16, CLIP_test_label17,
         CLIP_test_label18, CLIP_test_label19, CLIP_test_label20)

df_base <- inner_join(df_base, clip_df, by = "article_id")
cat("✅ Merged clip_text.csv\n")

# ============================
# 7. Merge CLIP_1 from 99_CLIP_all_0_shot.xlsx
# ============================
df_clip1 <- read_xlsx(file.path(result_folder, "99_CLIP_all_0_shot.xlsx")) %>%
  select(article_id, CLIP_1)
df_base <- inner_join(df_base, df_clip1, by = "article_id")
cat("✅ Merged CLIP_1\n")

# ============================
# 8. Save merged dataset
# ============================
write_csv(df_base, merged_output)
cat("🎉 DONE — Merged dataset saved to:", merged_output, "\n")
cat("📏 Final dataset dimensions:", paste(dim(df_base), collapse = " x "), "\n")

# ============================
# 9. Accuracy Calculation
# ============================
truth <- df_base$section_numeric

accuracy_df <- tibble(
  Model = colnames(df_base)[!(colnames(df_base) %in% c("article_id", "section_numeric"))],
  Accuracy = sapply(colnames(df_base)[!(colnames(df_base) %in% c("article_id", "section_numeric"))], function(col) {
    mean(df_base[[col]] == truth, na.rm = TRUE)
  })
)

# ============================
# 10. Save Accuracy Table
# ============================
write_csv(accuracy_df, accuracy_output)
cat("✅ Accuracy table saved to:", accuracy_output, "\n")
print(accuracy_df)


# ============================
# 11. Plot
# ============================
library(ggplot2)
library(dplyr)
library(readr)

# ========================
# ✅ MANUAL OFFSETS HERE
# ========================
offsets <- list(
  CLIP_0 = -0.007,
  GPT_5_nano_nr = -0.007,
  GPT_5_nano_r = -0.007,
  GPT_5_nr = -0.007,
  GPT_5_r = 0.007,
  GPT_41 = 0.007,
  BENCH1 = 0.005
)

# ======================================
# Load Accuracy Table
# ======================================
accuracy_df <- read_csv("Result/accuracy_df.csv")

# ======================================
# Extract CLIP baseline (0-shot)
# ======================================
clip0 <- accuracy_df %>% filter(Model == "CLIP_1") %>% pull(Accuracy)
y_min <- accuracy_df %>% filter(Model == "GPT_5_nano_nr") %>% pull(Accuracy) - 0.01

# ======================================
# Extract CLIP Models (Fine-tuned only)
# ======================================
clip_models <- accuracy_df %>%
  filter(grepl("CLIP_test_label", Model)) %>%
  arrange(factor(Model, levels = paste0("CLIP_test_label", 2:20))) %>%
  mutate(X_value = seq(2, 38, by = 2))

# ======================================
# GPT Baselines with Custom Offsets
# ======================================
gpt_labels <- tibble(
  Model = c("GPT_5_nano_nr", "GPT_5_nano_r", "GPT_5_nr", "GPT_5_r", "GPT_41"),
  Label = c("GPT-5-nano Non-Reasoning", "GPT-5-nano Reasoning",
            "GPT-5 Non-Reasoning", "GPT-5 Reasoning", "GPT-4.1")
)

gpt_lines <- accuracy_df %>%
  inner_join(gpt_labels, by = "Model") %>%
  mutate(
    LabelFull = paste0(Label, " (", sprintf("%.3f", Accuracy), ")"),
    Offset = c(
      offsets$GPT_5_nano_nr,
      offsets$GPT_5_nano_r,
      offsets$GPT_5_nr,
      offsets$GPT_5_r,
      offsets$GPT_41
    )
  ) %>%
  select(LabelFull, Accuracy, Offset)

# ======================================
# Single Benchmark (Blue)
# ======================================
benchmark1 <- 0.794  # ✅ Updated to your value

# ======================================
# PLOT
# ======================================
p <- ggplot(clip_models, aes(x = X_value, y = Accuracy)) +
  
  # === CLIP 0-shot baseline ===
  geom_hline(yintercept = clip0, linetype = "dashed", color = "grey50", size = 0.8) +
  annotate("text", x = max(clip_models$X_value) + 2, y = clip0 + offsets$CLIP_0,
           label = paste0("CLIP 0-shot (", sprintf("%.3f", clip0), ")"),
           hjust = 0, size = 3.5, color = "black") +
  
  # === GPT Baselines (Grey) ===
  geom_hline(data = gpt_lines, aes(yintercept = Accuracy), 
             linetype = "dashed", color = "grey50", size = 0.8) +
  geom_text(
    data = gpt_lines,
    aes(
      x = max(clip_models$X_value) + 2,
      y = Accuracy + Offset,
      label = LabelFull
    ),
    hjust = 0, size = 3.5, color = "black"
  ) +
  

  # Benchmark lines and annotations
  geom_hline(yintercept = 0.794, linetype = "dashed", color = "blue", linewidth = 0.6) +
  geom_hline(yintercept = 0.8333, linetype = "dashed", color = "blue", linewidth = 0.6) +
  annotate("text", x = Inf, y = 0.794, label = "                             Benchmark 1 (0.794)", hjust = 1.05, vjust = -0.4, size = 4, color = "blue") +
  annotate("text", x = Inf, y = 0.8333, label = "                             Benchmark 2 (0.833)", hjust = 1.05, vjust = -0.4, size = 4, color = "blue") +
  
  # === CLIP Fine-tune Curve ===
  geom_line(size = 1.2, color = "black") +
  geom_point(size = 2.8, color = "black") +
  
  # === AXIS + THEME ===
  scale_x_continuous(breaks = seq(0, 40, by = 4), limits = c(0, 55)) +
  scale_y_continuous(limits = c(0.6, 0.85)) +
  labs(x = "Number of Finetuned Data for CLIP (N = 100)", y = "Accuracy") +
  theme_classic(base_size = 14) +
  theme(
    axis.line = element_line(size = 0.8),
    axis.ticks = element_line(size = 0.6),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 110, 10, 10)
  )

# ======================================
# Save Plot
# ======================================
ggsave("Result_accuracy.png", plot = p, width = 10.1, height = 6, dpi = 300)


# ============================
# 12. Compute Accuracy + Macro/Micro F1, Precision, Recall for Each Model
# ============================

# ============================================================
# Compute Accuracy + Macro/Micro F1, Precision, Recall for Each Model
# ============================================================
library(MLmetrics)
library(caret)
library(dplyr)
library(tibble)
library(e1071)  # Required for confusionMatrix()

# ------------------------------------------------------------
# Rename CLIP columns to simpler names (CLIP_2, CLIP_3, ...)
# ------------------------------------------------------------
names(df_base) <- sub("^CLIP_test_label", "CLIP_", names(df_base))

# ------------------------------------------------------------
# Define ground truth and model columns
# ------------------------------------------------------------
truth <- df_base$section_numeric
model_cols <- colnames(df_base)[!(colnames(df_base) %in% c("article_id", "section_numeric"))]

# ------------------------------------------------------------
# Initialize list for results
# ------------------------------------------------------------
perf_list <- list()

# ------------------------------------------------------------
# Loop through each model and calculate metrics
# ------------------------------------------------------------
for (model in model_cols) {
  tryCatch({
    preds <- df_base[[model]]
    
    # Remove NAs
    valid_idx <- !is.na(preds) & !is.na(truth)
    y_true <- as.factor(truth[valid_idx])
    y_pred <- as.factor(preds[valid_idx])
    
    # Align factor levels between y_true and y_pred
    all_levels <- union(levels(y_true), levels(y_pred))
    y_true <- factor(y_true, levels = all_levels)
    y_pred <- factor(y_pred, levels = all_levels)
    
    # Compute Accuracy
    acc <- mean(y_true == y_pred)
    
    # Confusion Matrix
    cm <- confusionMatrix(y_pred, y_true)
    
    # ---------- Macro Metrics ----------
    precision_macro <- mean(cm$byClass[, "Precision"], na.rm = TRUE)
    recall_macro    <- mean(cm$byClass[, "Recall"], na.rm = TRUE)
    f1_macro        <- mean(cm$byClass[, "F1"], na.rm = TRUE)
    
    # ---------- Micro Metrics ----------
    tp <- sum(diag(cm$table))
    fp <- sum(colSums(cm$table)) - tp
    fn <- sum(rowSums(cm$table)) - tp
    
    precision_micro <- tp / (tp + fp)
    recall_micro    <- tp / (tp + fn)
    f1_micro        <- ifelse(
      (precision_micro + recall_micro) == 0,
      NA,
      2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    )
    
    # Store results
    perf_list[[model]] <- tibble(
      Model = model,
      Accuracy = acc,
      Macro_Precision = precision_macro,
      Macro_Recall = recall_macro,
      Macro_F1 = f1_macro,
      Micro_Precision = precision_micro,
      Micro_Recall = recall_micro,
      Micro_F1 = f1_micro
    )
  }, error = function(e) {
    message("⚠️ Error in model ", model, ": ", e$message)
  })
}

# ------------------------------------------------------------
# Combine all results into one dataframe
# ------------------------------------------------------------
perf_df <- bind_rows(perf_list)

# ------------------------------------------------------------
# View the results
# ------------------------------------------------------------
print(perf_df, n = Inf)

# ============================
# 13. Save CSV
# ============================
perf_output <- file.path(result_folder, "Model_Performance_Accuracy_F1.csv")
write_csv(perf_df, perf_output)

cat("✅ Accuracy + Macro F1 table saved to:", perf_output, "\n")
print(perf_df)
