require(rjags)

df <- read.csv("uefa_player_matches.csv", header = TRUE)
df <- df[complete.cases(df), ] # drop na
df <- df[(df$position %in% c("G", "D", "M", "F")),]
position_names <- list("G" = "Goalkeeper", "D" = "Defender", "M" = "Midfield", "F" = "Forward")
df$position <- factor(apply(df['position'], 1, function(x) position_names[[x]]))

y = apply(df['goals'], 1, function(x) if (x > 0) { 1 } else { 0 })
s = as.numeric(df$position)

Ntotal = length(y)
Nsubj = length(unique(s))

modelString = "
  model {
    for ( i in 1:Ntotal ) {
      y[i] ~ dbern( theta[s[i]] )
    }
    for ( sIdx in 1:Nsubj ) {
      theta[sIdx] ~ dbeta( omega*(kappa-2)+1 , (1-omega)*(kappa-2)+1 ) 
    }
    omega ~ dbeta( 1 , 1 ) # broad uniform
    kappa <- kappaMinusTwo + 2
    kappaMinusTwo ~ dgamma( 1.105125 , 0.1051249 )  # mode=1 , sd=10 
}
"
writeLines( modelString , con="TEMPmodel.txt" )

jagsModel = jags.model(
  "TEMPmodel.txt",
  data=list(
    y = y ,
    s = s ,
    Ntotal = Ntotal ,
    Nsubj = Nsubj
  )
)
update(jagsModel, n.iter=500) # burn-in

parameters = c("theta","omega","kappa") # The parameters to be monitored
codaSamples = coda.samples(jagsModel, variable.names=parameters, n.iter=10000)

mcmcMat = as.matrix(codaSamples)

for (i in 1:Nsubj) {
  position_name = levels(df$position)[i]
  position_mean = mean(mcmcMat[,paste("theta[", i, "]", sep="")])
  cat(paste(position_name, position_mean, "\n"))
}


