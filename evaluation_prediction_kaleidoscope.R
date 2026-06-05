kal = read.csv("kaleidoscope_classification_trained_with_full_dataset/output_advanced_classifier_on_test_set/cluster.csv")[-1] #ADJUST cluster.csv FOLDER
df = read.csv("runs/predictions_results.csv")[-1] ##ADJUST predictions_results.csv FOLDER
subtest=subset(df,subsample=="yes")
subtest$label=NA
subtest$label[subtest$Manual_ID=="manatee"] = 1
subtest$label[subtest$Manual_ID!="manatee"] = 0

kal$IN.FILE <- gsub("00033565_20250216T150000\\+0100_REC \\[-03.94513\\+011.34217\\]_loc1200-1251.09065759637s.wav",
                    "00033565_20250216T150000+0100_REC_[-03.94513+011.34217]_loc1200-1260s.wav",
                    kal$IN.FILE)
kal$IN.FILE <- gsub("_REC_", "_", kal$IN.FILE)
kal$IN.FILE <- gsub("_REC ", "_", kal$IN.FILE)
kal$IN.FILE <- gsub("_Rec_", "_", kal$IN.FILE)
kal$IN.FILE <- gsub("_Rec ", "_", kal$IN.FILE)

colnames(kal)[2]="Fichier"
kal$Fichier=substr(kal$Fichier, 1, nchar(kal$Fichier) - 4)
kal_sub = kal[kal$TOP1MATCH.=="Lamantin",]

test <- subtest %>%
  inner_join(kal,by = "Fichier") %>%                     # associer par fichier
  mutate(diff = abs(Localisation - OFFSET)) %>%                   # calculer l'écart
  group_by(Fichier, Localisation) %>%                            # pour chaque loc d’un fichier
  slice_min(order_by = diff, n = 1, with_ties = FALSE) %>%  # garder le plus proche
  subset(diff<max(kal_sub$DURATION)) %>%
  ungroup() %>%
  group_by(OFFSET) %>%
  slice_min(order_by=diff,n=1) %>%
  ungroup()

summary(as.factor(test$MANUAL.ID))

subtest = merge(subtest,test[c(1,3,20:46)], by=c("Fichier","Localisation"),all.x=T)
subtest$true_false_kal = NA

for (i in 1:nrow(subtest)){
  if (isTRUE(subtest$TOP1MATCH.[i]=="Lamantin")){
    if (subtest$Manual_ID[i]=="manatee"){
      subtest$true_false_kal[i] = "TP"
    }
    else{
      subtest$true_false_kal[i] = "FP"
    }
  }
  else{
    if (subtest$Manual_ID[i]=="manatee"){
      subtest$true_false_kal[i]= "FN"
    }
    else{
      subtest$true_false_kal[i]="TN"
    }
  }
}

subtest$true_false_kal =as.factor(subtest$true_false_kal)
summary(subtest$true_false_kal)
check = subtest[,c("label","true_false_kal","TOP1MATCH.")]
df=subset(subtest,!is.na(true_false_kal))
results = data.frame()

TP = sum(df[["true_false_kal"]] == "TP", na.rm = TRUE)
TN = sum(df[["true_false_kal"]] == "TN", na.rm = TRUE)
FP = sum(df[["true_false_kal"]] == "FP", na.rm = TRUE)
FN = sum(df[["true_false_kal"]] == "FN", na.rm = TRUE)

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
    title = "Kaleidoscope evaluation metrics",
    x = "Metric",
    y = "Value"
  ) +
  theme_minimal()
