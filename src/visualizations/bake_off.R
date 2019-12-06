library(tidyverse)

score <- data.frame(
  model = c('Logistic SGD', 
            'Logistic SAG', 
            'Huber SGD', 
            'LDA', 
            'XGBoost', 
            'CatBoost', 
            'LightGBM'),
  log_loss = c(2.63452, 
               2.51241, 
               3.62403, 
               2.53357, 
               2.24266, 
               2.27116, 
               2.27000),
  val_log_loss = c(2.62686,
                   2.51170,
                   3.62533,
                   2.53265,
                   2.23738,
                   2.19239,
                   2.25000),
  time = c(18.36,
           72.44,
           18.39,
           25,
           16648.21,
           16200,
           10000)
)

ggplot(data = score) + 
  geom_bar(
    mapping = aes(
      x = reorder(model, -log_loss), 
      y = log_loss, 
      fill = log(time)
    ), 
    stat = 'identity'
  ) +
  coord_flip() +
  labs(
    x = '', 
    y = ''
  ) +
  theme_economist() +
  scale_color_economist() +
  theme(
    #    legend.position = 'none',
    axis.text = element_text(size = 30)
  )
  
  # +
  # scale_fill_gradient(low = 'lightblue', high = 'darkcyan',
  #                     space = "Lab", na.value = "grey50", guide = "colourbar",
  #                     aesthetics = "fill")           
ggplot(data = score, 
       mapping = aes(
         x = log_loss,
         y = time)
       ) + 
  geom_point()
