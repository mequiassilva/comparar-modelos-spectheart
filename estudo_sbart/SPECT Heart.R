library(SoftBart)
library(BART)          # usar pbart em vez de dbarts::bart
library(randomForest)
library(xgboost)
library(ggplot2)
library(pROC)
library(dplyr)
library(tidyr)
library(writexl)

# ---- Função de métricas ----
calc_metrics <- function(y_true, prob, pred) {
  # Caso 1: só uma classe em y_true
  if(length(unique(y_true)) < 2) {
    return(data.frame(
      Acuracia = mean(y_true == pred),
      AUC = NA,
      Sensibilidade = NA,
      Especificidade = NA,
      F1_score = NA
    ))
  }
  
  # Caso 2: probabilidade constante
  if(length(unique(prob)) < 2) {
    return(data.frame(
      Acuracia = mean(y_true == pred),
      AUC = NA,
      Sensibilidade = sum(y_true == 1 & pred == 1) / sum(y_true == 1),
      Especificidade = sum(y_true == 0 & pred == 0) / sum(y_true == 0),
      F1_score = 2 * sum(y_true == 1 & pred == 1) / (sum(pred == 1) + sum(y_true == 1))
    ))
  }
  
  # Caso normal
  roc_obj <- roc(y_true, prob)
  data.frame(
    Acuracia = mean(y_true == pred),
    AUC = as.numeric(auc(roc_obj)),
    Sensibilidade = sum(y_true == 1 & pred == 1) / sum(y_true == 1),
    Especificidade = sum(y_true == 0 & pred == 0) / sum(y_true == 0),
    F1_score = 2 * sum(y_true == 1 & pred == 1) / (sum(pred == 1) + sum(y_true == 1))
  )
}


# Estudando os dados
list.files()
file.info(c("SPECT.train", "SPECT.test", "SPECT.names"))

# Procurar as partes do arquivo .names que falam
# sobre os atributos e a variável de classificação

grep(
  "attribute|Attribute|class|Class|diagnosis|Diagnosis|feature|Feature",
  readLines("SPECT.names"),
  value = TRUE
)

# Mostrar as últimas linhas do arquivo .names
tail(readLines("SPECT.names"), 30)

# ---- Carregar a base SPECT ----

SPECT_train <- read.csv(
  "SPECT.train",
  header = FALSE
)

SPECT_test <- read.csv(
  "SPECT.test",
  header = FALSE
)

# ---- Nomear as 23 colunas ----

nomes <- c(
  "OVERALL_DIAGNOSIS",
  paste0("F", 1:22)
)

colnames(SPECT_train) <- nomes
colnames(SPECT_test) <- nomes

# ---- Conferência da estrutura ----

dim(SPECT_train)
dim(SPECT_test)

head(SPECT_train)
head(SPECT_test)

str(SPECT_train)
str(SPECT_test)


# ---- Preparação dos dados para os modelos ----

Xtrain <- as.matrix(SPECT_train[, -1])
Ytrain <- SPECT_train$OVERALL_DIAGNOSIS

Xtest <- as.matrix(SPECT_test[, -1])
Ytest <- SPECT_test$OVERALL_DIAGNOSIS

colnames(Xtrain) <- paste0("F", 1:22)
colnames(Xtest)  <- paste0("F", 1:22)

# ---- SoftBART ----
# set.seed()
set.seed(2006) #para métricas 
# set.seed(5050) #para importancias 
sbart_fit <- softbart(
  X = Xtrain,
  Y = Ytrain,
  X_test = Xtest
)

prob_sbart <- sbart_fit$y_hat_test_mean

pred_sbart <- ifelse(prob_sbart > 0.5, 1, 0)

# ---- Previsões do SoftBART ----

head(prob_sbart)
head(pred_sbart)

table(pred_sbart)

# ---- Métricas do SoftBART ----

metricas_sbart <- calc_metrics(
  Ytest,
  prob_sbart,
  pred_sbart
)

metricas_sbart

# ---- BART ----

bart_fit <- pbart(
  x.train = Xtrain,
  y.train = Ytrain,
  x.test = Xtest,
  sparse = TRUE
)

prob_bart <- bart_fit$prob.test.mean

pred_bart <- ifelse(prob_bart > 0.5, 1, 0)

# Métricas
metricas_bart <- calc_metrics(
  Ytest,
  prob_bart,
  pred_bart
)

metricas_bart

# ---- Random Forest ----

rf_fit <- randomForest(
  as.factor(Ytrain) ~ .,
  data = data.frame(Xtrain),
  ntree = 300,
  mtry = 2,
  maxnodes = 10,
  nodesize = 20
)

prob_rf <- predict(
  rf_fit,
  newdata = data.frame(Xtest),
  type = "prob"
)[, 2]

pred_rf <- ifelse(prob_rf > 0.5, 1, 0)

# Métricas
metricas_rf <- calc_metrics(
  Ytest,
  prob_rf,
  pred_rf
)

metricas_rf

# ---- XGBoost ----

dtrain <- xgb.DMatrix(
  data = Xtrain,
  label = Ytrain
)

dtest <- xgb.DMatrix(
  data = Xtest,
  label = Ytest
)

params <- list(
  objective = "binary:logistic"
)

xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300
)

# Probabilidades
prob_xgb <- predict(
  xgb_fit,
  dtest
)

# Classificação usando ponto de corte 0.5
pred_xgb <- ifelse(
  prob_xgb > 0.5,
  1,
  0
)

# Métricas
metricas_xgb <- calc_metrics(
  Ytest,
  prob_xgb,
  pred_xgb
)

metricas_xgb

# ---- Comparação dos modelos ----

comparacao_modelos <- bind_rows(
  metricas_sbart %>% mutate(Modelo = "SoftBART"),
  metricas_bart  %>% mutate(Modelo = "BART"),
  metricas_rf    %>% mutate(Modelo = "Random Forest"),
  metricas_xgb   %>% mutate(Modelo = "XGBoost")
) %>%
  select(
    Modelo,
    Acuracia,
    AUC,
    Sensibilidade,
    Especificidade,
    F1_score
  )

comparacao_modelos

# ============================================================
# IMPORTÂNCIA DAS VARIÁVEIS - SPECT
# ============================================================

# ---- SoftBART ----
importancia_sbart_df <- data.frame(
  Variavel = paste0("F", 1:22),
  Importancia = colMeans(sbart_fit$var_counts)
)

# ---- BART ----
importancia_bart_df <- data.frame(
  Variavel = paste0("F", 1:22),
  Importancia = as.numeric(bart_fit$varcount.mean)
)

# ---- Random Forest ----
imp_rf <- importance(rf_fit, type = 2)

importancia_rf_df <- data.frame(
  Variavel = rownames(imp_rf),
  Importancia = as.numeric(imp_rf[, 1]),
  row.names = NULL
)

# ---- XGBoost ----
imp_xgb <- xgb.importance(model = xgb_fit)

importancia_xgb_df <- data.frame(
  Variavel = paste0("F", 1:22)
) %>%
  left_join(
    imp_xgb[, c("Feature", "Gain")],
    by = c("Variavel" = "Feature")
  ) %>%
  mutate(
    Importancia = ifelse(is.na(Gain), 0, Gain)
  ) %>%
  select(Variavel, Importancia)

# ============================================================
# CONFERÊNCIA
# ============================================================

cat("SoftBART:", nrow(importancia_sbart_df), "\n")
cat("BART:", nrow(importancia_bart_df), "\n")
cat("Random Forest:", nrow(importancia_rf_df), "\n")
cat("XGBoost:", nrow(importancia_xgb_df), "\n")

importancias <- bind_rows(
  SoftBART = importancia_sbart_df,
  BART = importancia_bart_df,
  `Random Forest` = importancia_rf_df,
  XGBoost = importancia_xgb_df,
  .id = "Modelo"
)

importancias_norm <- importancias %>%
  group_by(Modelo) %>%
  mutate(
    Importancia_norm = Importancia / max(Importancia)
  ) %>%
  ungroup()


importancias_norm$Variavel <- factor(
  importancias_norm$Variavel,
  levels = paste0("F", 1:22),
  ordered = TRUE
)

# ============================================================
# GRÁFICO - COMPARAÇÃO DAS MÉTRICAS DOS MODELOS
# ============================================================

# Identidade visual do pôster
cores <- c(
  "SoftBART" = "#073A70",
  "BART" = "#6B7280",
  "Random Forest" = "#A7AFB8",
  "XGBoost" = "#F9BD38"
)

# Organizar as métricas em formato longo
metricas_long <- comparacao_modelos %>%
  pivot_longer(
    cols = c(
      Acuracia,
      Sensibilidade,
      Especificidade,
      F1_score,
      AUC
    ),
    names_to = "Metrica",
    values_to = "Valor"
  ) %>%
  mutate(
    Modelo = factor(
      Modelo,
      levels = c(
        "SoftBART",
        "BART",
        "Random Forest",
        "XGBoost"
      )
    ),
    Metrica = factor(
      Metrica,
      levels = c(
        "Acuracia",
        "Sensibilidade",
        "Especificidade",
        "F1_score",
        "AUC"
      ),
      labels = c(
        "Acurácia",
        "Sensibilidade",
        "Especificidade",
        "F1-score",
        "AUC"
      )
    )
  )

# Criar gráfico
grafico_metricas <- ggplot(
  metricas_long,
  aes(
    x = Metrica,
    y = Valor,
    fill = Modelo
  )
) +
  geom_col(
    position = position_dodge(width = 0.95),
    width = 0.75
  ) +
  geom_text(
    aes(label = sprintf("%.2f", Valor)),
    position = position_dodge(width = 0.95),
    vjust = -0.4,
    size = 3.8
  ) +
  scale_fill_manual(
    values = cores,
    breaks = c(
      "SoftBART",
      "BART",
      "Random Forest",
      "XGBoost"
    ),
    labels = c(
      "SBART",
      "BART",
      "Floresta Aleatória",
      "XGBoost"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1.10),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "",
    x = "Métrica de avaliação",
    y = "Valor",
    fill = "Modelo"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 16,
      hjust = 0.5
    ),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(
      size = 12
    ),
    axis.text.y = element_text(size = 12),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# Exibir gráfico
print(grafico_metricas)

ggsave(
  "desempenho_modelos_spect.png",
  plot = grafico_metricas,
  width = 20,
  height = 10,
  units = "cm",
  dpi = 600,
  bg = "white"
)

# ============================================================
# GRÁFICO - IMPORTÂNCIA RELATIVA DAS VARIÁVEIS
# ============================================================

library(ggplot2)
library(dplyr)

# Identidade visual do pôster
cores <- c(
  "SoftBART" = "#073A70",
  "BART" = "#6B7280",
  "Random Forest" = "#A7AFB8",
  "XGBoost" = "#F9BD38"
)

# Definir a ordem dos modelos
importancias_norm$Modelo <- factor(
  importancias_norm$Modelo,
  levels = c(
    "SoftBART",
    "BART",
    "Random Forest",
    "XGBoost"
  )
)

# Garantir a ordem das variáveis F1 a F22
importancias_norm$Variavel <- factor(
  importancias_norm$Variavel,
  levels = paste0("F", 1:22),
  ordered = TRUE
)

# Criar gráfico
grafico_importancia <- ggplot(
  importancias_norm,
  aes(
    x = Variavel,
    y = Importancia_norm,
    fill = Modelo
  )
) +
  geom_col(
    position = position_dodge(width = 0.85),
    width = 0.8
  ) +
  scale_fill_manual(
    values = cores,
    breaks = c(
      "SoftBART",
      "BART",
      "Random Forest",
      "XGBoost"
    ),
    labels = c(
      "SBART",
      "BART",
      "Floresta Aleatória",
      "XGBoost"
    ),
    drop = FALSE
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "",
    x = "Variável",
    y = "Importância relativa",
    fill = "Modelo"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 16,
      hjust = 0.5
    ),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      size = 12.9
    ),
    axis.text.y = element_text(size = 14),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# Exibir gráfico
print(grafico_importancia)

# ============================================================
# SALVAR EM ALTA QUALIDADE
# ============================================================

ggsave(
  filename = "importancia_variaveis_spect.png",
  plot = grafico_importancia,
  width = 20,
  height = 10,
  units = "cm",
  dpi = 600,
  bg = "white"
)

