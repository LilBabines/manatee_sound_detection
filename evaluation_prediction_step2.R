library(tidyr)

df = read.csv("runs/predictions_results.csv")[-1]
subtest=subset(df,subsample=="yes")
subtest$label=NA
subtest$label[subtest$Manual_ID=="manatee"] = 1
subtest$label[subtest$Manual_ID!="manatee"] = 0

subtest$true_false = NA

for (i in 1:nrow(subtest)){
  if (subtest$label[i] == 1){
    if (subtest$id_pred_1[i] == 1 || isTRUE(subtest$id_pred_2[i] == 1)) {
      subtest$true_false[i] <- "TP"
    } else {
      subtest$true_false[i] <- "FN"
    }
  }  else {
    if (subtest$id_pred_1[i] == 0 && (subtest$id_pred_2[i] == 0 || is.na(subtest$id_pred_2[i]))) {
      subtest$true_false[i] <- "TN"
    } else {
      subtest$true_false[i] <- "FP"
    }
  }
}

result <- subtest %>%
  mutate(idx = row_number()) %>%                # indice global de chaque ligne
  arrange(Fichier, Localisation) %>%
  group_by(Fichier) %>%
  mutate(next_loc   = lead(Localisation),
         next_label = lead(label),
         next_idx   = lead(idx)) %>%            # indice de la ligne suivante
  filter(label != next_label) %>%               # seulement les labels contradictoires
  mutate(ecart = next_loc - Localisation) %>%
  ungroup() %>%
  select(Fichier, idx, next_idx, Localisation, next_loc, label, next_label, ecart)

for (i in 1: nrow(result)){
  if (result$ecart[i]<=10){
    if (result$label[i]==0){
      subtest$true_false[result$idx[i]] = NA
    }
    else {
      subtest$true_false[result$next_idx[i]] = NA
    }
  }
}

subtest$true_false =as.factor(subtest$true_false)
summary(subtest$true_false)

df=subset(subtest,!is.na(true_false))
results = data.frame()

TP = sum(df[["true_false"]] == "TP", na.rm = TRUE)
TN = sum(df[["true_false"]] == "TN", na.rm = TRUE)
FP = sum(df[["true_false"]] == "FP", na.rm = TRUE)
FN = sum(df[["true_false"]] == "FN", na.rm = TRUE)
  
acc <- (TP + TN) / (TP + TN + FP + FN)
prec <- TP/(TP+FP)
recall <- TP/(TP+FN)
F1 = (2*prec*recall)/(prec+recall)
  
results <- rbind(results, data.frame(accuracy = acc,precision=prec,recall=recall,F1_score=F1))

plot_df <- results %>%
  pivot_longer(cols = c(accuracy, precision, recall, F1_score),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(metric,
                    levels = c("accuracy","precision","recall","F1_score"),
                    labels = c("Accuracy","Precision","Recall","F1-score"))
  )

# Display plot_df as a table
print(plot_df)

# Display plot_df in the RStudio Viewer
View(plot_df)

# Plot the metrics
ggplot(plot_df, aes(x = metric, y = value)) +
  geom_col() +
  geom_text(
    aes(label = round(value, 3)),
    vjust = -0.5,
    size = 4
  ) +
  ylim(0, 1) +
  labs(
    title = "Model evaluation metrics",
    x = "Metric",
    y = "Value"
  ) +
  theme_minimal()
