required_packages = c("tidyr", "here", "nloptr", "lme4", "tibble", "optparse","tidyverse","dplyr","simr", "purrr", "jsonlite")
invisible(lapply(required_packages, require, character.only = TRUE))


# Function to calculate Euclidean distance
euclidean_distance <- function(coord, pos) {
  if (length(coord) != 2 || length(pos) != 2) return(NA)  # (handles NA cases)
  sqrt((coord[1] - pos[1])^2 + (coord[2] - pos[2])^2)
}


# Read in data
df1 <- read.csv(file.path("powerdata", "alldata.csv"), header = TRUE) 

#Exclude bad ppt. based on infrequency items  
dfquest <- read.csv(file.path("powerdata/questionnaires", "processedsurvey.csv"), header = TRUE) 
dfquest$ID = dfquest$PROLIFICID
df1 <- df1 %>%
  left_join(dfquest, by = "ID")

failed_checks <- unique(df1$ID[df1$checks == 0])
df1 <- df1[!df1$ID %in% failed_checks, ]
length(unique(df1$ID))

#weird ppt exclude
weird <- c("46489216914439", "355538383730854")
df1 <- df1[!df1$ID %in% weird, ]

df1 <- df1 %>% mutate(
  csPosition = map(csPosition, ~ if (is.character(.x)) fromJSON(.x) else .x),
  ratingCoordinate = map(ratingCoordinate, ~ if (is.character(.x)) fromJSON(.x) else .x)
)

df1 <- df1 %>%
  mutate(csX = map_dbl(csPosition, 1), 
         csY = map_dbl(csPosition, 2)) 

df <- df1 %>%
  mutate(
    rateX = map_dbl(ratingCoordinate, ~ if (length(.x) == 2) .x[[1]] else 0),
    rateY = map_dbl(ratingCoordinate, ~ if (length(.x) == 2) .x[[2]] else 0)
  )

#adjust for 1 indexing 
df$csX = df$csX+1
df$csY = df$csY+1
df$rateX = df$rateX+1
df$rateY = df$rateY+1
length(unique(df$ID))

#expand the navi section into 1 move x row 
df <- df %>%
  mutate(
    path_clean = gsub("'", "\"", path),
    moveTime_clean = gsub("'", "\"", moveTime)
  )

navigate_df <- df %>% filter(state == "navigating")

expanded_df <- navigate_df %>%
  mutate(
    parsed_path = map(path_clean, ~ fromJSON(.x)),
    parsed_time = map(moveTime_clean, ~ fromJSON(.x))
  ) %>%
  unnest_longer(parsed_path, indices_to = "move") %>%
  mutate(
    move_col = map_dbl(parsed_path, "col"),
    move_row = map_dbl(parsed_path, "row"),
    move_time = map2_dbl(parsed_time, move, ~ .x[[.y]])
  ) %>%
  select(-parsed_path, -parsed_time) %>%
  mutate(move = as.character(move))

non_navigate <- df %>% filter(state != "navigating")

non_navigate <- non_navigate %>%
  mutate(
    move = NA_character_,
    move_col = NA_real_,
    move_row = NA_real_,
    move_time = NA_real_
  )

final_df <- bind_rows(expanded_df, non_navigate) 

final_df$move_col = final_df$move_col +1
final_df$move_row = final_df$move_row +1

final_df <- final_df %>%
  mutate(distance = map2_dbl(ratingCoordinate, csPosition, euclidean_distance))


#Fit visits ~ rating*anxiety model

# change ratings
df_pre_avg <- final_df %>%
  filter(state == "ratingPRE") %>%
  group_by(ID, run, distance, rateX, rateY, sticsa_score, date) %>%
  summarise(ratingPRE = mean(rating, na.rm = TRUE), .groups = "drop")

df_by_ppt <- final_df %>%
  filter(state == "navigating") %>%
  mutate(
    distance = round(sqrt((move_row - csX)^2 + (move_col - csY)^2))
  ) %>%
  group_by(ID, move_col, move_row, run) %>%
  summarise(visits = n(), .groups = "drop")

df_by_ppt$rateX <- df_by_ppt$move_row
df_by_ppt$rateY <- df_by_ppt$move_col
joint_df <- left_join(df_pre_avg, df_by_ppt, by = c("ID", "rateX", "rateY", "run"))
joint_df$visits[is.na(joint_df$visits)] <- 0 #assign 0 to never visited


#add anx 
joint_df$anx_z <- (joint_df$sticsa_score-mean(joint_df$sticsa_score))/sd(joint_df$sticsa_score)
joint_df$distance2 <- joint_df$distance^2 

set.seed(123)
n_boot <- 4
results_list <- vector("list", n_boot)

for (i in 1:n_boot) {
  message("Bootstrap iteration ", i)
  
  # rand sample 80% of ppt. 
  all_ids <- unique(joint_df$ID)
  sampled_ids <- sample(all_ids, size = floor(0.8 * length(all_ids)), replace = FALSE)
  boot_df <- joint_df[joint_df$ID %in% sampled_ids, ]
  
  print(length(unique(boot_df$ID)))
  
  # Fit model & get effects 
  m.quad1 <- lme4::lmer(data = boot_df, 
                  visits ~ ratingPRE*anx_z + distance*anx_z + distance2*anx_z + (1 | ID))
  
  ccc_a <- as.data.frame(summary(m.quad1)$coefficients)
  fixed_a <- ccc_a$Estimate
  
  # Get var components
  rand <- as.numeric(VarCorr(m.quad1)$ID)
  rand <- matrix(rand, nrow = 1)
  res <- sigma(m.quad1)
  
  # Sim data
  n = 300 
  IDs <- factor(1:n)
  runs <- 0:8
  ratingDistances <- c(1, 2, 3, 5)
  
  sim_data <- expand.grid(ID = IDs,
                          run = runs,
                          ratingDistance = ratingDistances)
  
  # Assign means for each ratingDistance group based on 
  sim_data$mean_pre <- NA
  
  # Get mean & sd for each distance & sim 
  mean_sd_by_dist <- boot_df %>%
    group_by(distance) %>%
    summarise(
      mean_val = mean(ratingPRE, na.rm = TRUE),
      sd_val   = sd(ratingPRE, na.rm = TRUE),
      .groups = "drop"
    )
  
  mean_sd_by_dist$ratingDistance = mean_sd_by_dist$distance
  sim_data <- sim_data %>%
    left_join(mean_sd_by_dist, by = "ratingDistance") %>%
    rowwise() %>%
    mutate(mean_pre = rnorm(1, mean = mean_val, sd = sd_val)) %>%
    ungroup() %>%
    select(-mean_val, -sd_val)
  
  #sample anx scores from real data to match dist
  sim_data$anx_z <- sample(boot_df$anx_z, size = nrow(sim_data), replace = TRUE)
  sim_data$ratingDistance2 <- sim_data$ratingDistance^2
  sim_data$ratingDistance2 <- sim_data$ratingDistance^2
  
  # Make Lmer model with simr
  model_add <- simr::makeLmer(visits ~ mean_pre*anx_z + ratingDistance*anx_z + ratingDistance2*anx_z + (1|ID), 
                        fixef = fixed_a, VarCorr = rand, sigma = res, data = sim_data)
  
  # PowerSim on sim data 
  sim_q3 <- simr::powerSim(model_add, nsim = 100, 
                     test = simr::fcompare(~ mean_pre*anx_z + ratingDistance + ratingDistance2 + anx_z))
  
  # PowerCurve on sim data 
  p_curve_q3 <- simr::powerCurve(model_add, 
                           test = simr::fcompare(~ mean_pre*anx_z + ratingDistance + ratingDistance2 + anx_z), 
                           along = "ID", nsim = 100)
  df_power <- summary(p_curve_q3)
  
  # save
  results_list[[i]] <- list(
    iteration = i,
    fixed_effects = fixed_a,
    power_sim = summary(sim_q3),
    power_curve = df_power
  )
}


fixed_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(iteration = x$iteration, t(x$fixed_effects))
}))

power_df <- do.call(rbind, lapply(results_list, function(x) {
  cbind(iteration = x$iteration, x$power_curve)
}))


#write.csv(power_df, file.path("powersim/poweroutput", "powercurves_quadr.csv"), row.names = FALSE)
