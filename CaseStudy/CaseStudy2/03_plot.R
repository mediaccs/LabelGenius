# ============================
# Load libraries
# ============================
library(ggplot2)
library(dplyr)
library(readr)

# ============================
# Load Data
# ============================
df <- read_csv("Result/02_performance.csv", show_col_types = FALSE)

# ============================
# Apply Shortened Labels
# ============================
df$Model <- case_when(
  df$Model == "GPT_41" ~ "GPT-4.1 (0-shot)",
  df$Model == "GPT_41_finetune_1" ~ "GPT-4.1 (200)",
  df$Model == "GPT_41_finetune_2" ~ "GPT-4.1 (400)",
  df$Model == "GPT_41_finetune_3" ~ "GPT-4.1 (600)",
  df$Model == "GPT_41_finetune_4" ~ "GPT-4.1 (800)",
  df$Model == "GPT_41_finetune_5" ~ "GPT-4.1 (1000)",
  df$Model == "GPT_5_nano_nr" ~ "GPT-5-nano (NR)",
  df$Model == "GPT_5_nano_r" ~ "GPT-5-nano (R)",
  df$Model == "GPT_5_nr" ~ "GPT-5 (NR)",
  df$Model == "GPT_5_r" ~ "GPT-5 (R)",
  TRUE ~ df$Model
)

# ============================
# Legend Order
# ============================
legend_levels_exact <- c(
  "GPT-4.1 (0-shot)",
  "GPT-4.1 (200)",
  "GPT-4.1 (400)",
  "GPT-4.1 (600)",
  "GPT-4.1 (800)",
  "GPT-4.1 (1000)",
  "GPT-5-nano (NR)",
  "GPT-5-nano (R)",
  "GPT-5 (NR)",
  "GPT-5 (R)"
)

df$Model <- factor(df$Model, levels = legend_levels_exact)

# ============================
# Theme Labels
# ============================
theme_labels <- c(
  "Q3_1" = "Economic consequences",
  "Q3_2" = "Crime/safety",
  "Q3_3" = "Family",
  "Q3_4" = "Immigrant wellbeing",
  "Q3_5" = "Culture/society",
  "Q3_6" = "Politics",
  "Q3_7" = "Legislation/regulation",
  "Q3_8" = "Public opinion"
)
df$Theme <- factor(df$Theme, levels = names(theme_labels), labels = theme_labels)

# ============================
# Color Palette (Grouped Gradient)
# ============================
model_palette <- c(
  "GPT-4.1 (0-shot)" = "#E0E0E0",
  "GPT-4.1 (200)" = "#C0C0C0",
  "GPT-4.1 (400)" = "#A0A0A0",
  "GPT-4.1 (600)" = "#808080",
  "GPT-4.1 (800)" = "#505050",
  "GPT-4.1 (1000)" = "#000000",
  "GPT-5-nano (NR)" = "#4A90E2",
  "GPT-5-nano (R)" = "#003366",
  "GPT-5 (NR)" = "#6AA84F",
  "GPT-5 (R)" = "#274E13"
)

# ============================
# Global Y Zoom + Baseline Line
# ============================
range_y <- range(df$Accuracy, na.rm = TRUE)
buffer <- 0.02
y_min_global <- max(range_y[1] - buffer, 0)

# ============================
# Plot
# ============================
# ============================
# Plot with Benchmark Background
# ============================
p <- ggplot(df, aes(x = Model, y = Accuracy, color = Model)) +
  # Benchmark background band
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = 0.83, ymax = 0.95),
            fill = "#F0F8FF", alpha = 0.4, inherit.aes = FALSE) +
  geom_hline(yintercept = y_min_global, linewidth = 0.4, alpha = 0.6) +
  geom_point(size = 4, alpha = 0.95) +
  scale_color_manual(values = model_palette) +
  scale_y_continuous(limits = c(0.6, 1)) +
  facet_wrap(~ Theme, ncol = 4, strip.position = "bottom") +
  labs(x = NULL, y = NULL, color = NULL) +
  theme_void(base_size = 20) +
  theme(
    axis.line.y = element_line(color = "black"),
    axis.text.y = element_text(size = 14),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    strip.text = element_text(size = 18, face = "bold", margin = margin(t = 6, b = 6)),
    panel.spacing = unit(1.2, "lines"),
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    plot.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA)
  )


# ============================
# Save
# ============================
ggsave("Result/plot_accuracy.pdf", plot = p, width = 16, height = 10, dpi = 300, bg = "transparent")
ggsave("Result/plot_accuracy.png", plot = p, width = 16, height = 10, dpi = 300, bg = "transparent")

print("EXPORT COMPLETE: COMPACT LEGEND VERSION SAVED")
