require(rjags)

df <- read.csv("uefa_player_matches.csv", header = TRUE)
df <- df[complete.cases(df), ] # drop na
df <- df[(df$position %in% c("G", "D", "M", "F")),]
position_names <- list("G" = "Goalkeeper", "D" = "Defender", "M" = "Midfield", "F" = "Forward")
df$position <- factor(apply(df['position'], 1, function(x) position_names[[x]]))
df$player_id <- factor(df$player_id)

y = apply(df['goals'], 1, function(x) if (x > 0) { 1 } else { 0 })
positions = as.numeric(df$position)
playerIds = as.numeric(df$player_id)

playersMostCommonPositions = c()
for (playerId in 1:length(playerIds)) {
  playerPositions = df[(playerIds == playerId),]
  mostCommon = sort(playerPositions$position, decreasing=FALSE)[1]
  playersMostCommonPositions <- c(playersMostCommonPositions, mostCommon)
}

lengthY = length(y)
lengthPositions = length(unique(positions))
lengthPlayerIds = length(unique(playerIds))

modelString = "
  model {
    kappaMinusTwo ~ dgamma( 1.105125 , 0.1051249 )  # mode=1 , sd=10 
    kappa <- kappaMinusTwo + 2

    omega ~ dbeta( 1 , 1 ) # broad uniform

    for ( positionId in 1:lengthPositions ) {
      theta[positionId] ~ dbeta(omega * (kappa-2) + 1, (1 - omega) * (kappa - 2) + 1) 
    }

    kappaMinusTwoPlayer ~ dgamma( 1.105125 , 0.1051249 )  # mode=1 , sd=10 
    kappaPlayer <- kappaMinusTwoPlayer + 2

    for ( playerId in 1:lengthPlayerIds ) {
      omegaPlayer[playerId] = theta[playersMostCommonPositions[playerId]]
      thetaPlayer[playerId] ~ dbeta(
        omegaPlayer[playerId] * (kappaPlayer - 2) + 1, (1 - omegaPlayer[playerId]) * (kappaPlayer-2) + 1
      )
    }

    for ( i in 1:lengthY ) {
      y[i] ~ dbern(thetaPlayer[playerIds[i]])
    }
  }
"
modelString = "
  model {
    for (positionId in 1:lengthPositions) {
      meanPosition[positionId] ~ dunif(0, 1)
      variancePosition[positionId] ~ dunif(0, 1)
    }
    for (i in 1:lengthPlayerIds) {
      # Creating a beta distribution based on mean and variance: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
      mean[i] = meanPosition[playersMostCommonPositions[i]]
      variance[i] = variancePosition[playersMostCommonPositions[i]] / 10
      v[i] = ((mean[i] * (1 - mean[i])) / variance[i]) - 1
      alpha[i] = mean[i] * v[i]
      beta[i] = (1 - mean[i]) * v[i]
    
      probPlayer[i] ~ dbeta(alpha[i], beta[i])
    }
    for (i in 1:lengthY) {
      y[i] ~ dbern(probPlayer[playerIds[i]])
    }
  }
"

writeLines(modelString, con="TEMPmodel.txt")

jagsModel = jags.model(
  "TEMPmodel.txt",
  data=list(
    y = y,
    playerIds = playerIds,
    playersMostCommonPositions = playersMostCommonPositions,
    lengthY = lengthY,
    lengthPositions = lengthPositions,
    lengthPlayerIds = lengthPlayerIds
  )
)
update(jagsModel, n.iter=500) # burn-in

# parameters = c("theta","omega","kappa","kappaPlayer","thetaPlayer") # The parameters to be monitored
parameters = c("probPlayer", "meanPosition", "variancePosition")
codaSamples = coda.samples(jagsModel, variable.names=parameters, n.iter=10000)

mcmcMat = as.matrix(codaSamples)

install.packages("bayesboot")
require(bayesboot)


for (i in 1:lengthPositions) {
  position_name = levels(df$position)[i]
  print(position_name)
  positionMeans = mcmcMat[,paste("meanPosition[", i, "]", sep="")]
  varianceMeans = mcmcMat[,paste("variancePosition[", i, "]", sep="")]
  plotHDI(positionMeans, credMass=0.95, Title=paste(position_name, "HDI"))
  print(quantile(positionMeans))
  print(quantile(varianceMeans))
}

positionLevel = which(levels(df$position) == "Forward")
positionMeans = mcmcMat[,paste("meanPosition[", positionLevel, "]", sep="")]
plotPost(positionMeans, credMass=0.95, xlab="Forward mean HDI")

varianceMeans = mcmcMat[,paste("variancePosition[", positionLevel, "]", sep="")]
plotPost(varianceMeans, credMass=0.95, xlab="Forward variance HDI")

neymarId = 276
neymarLevel = which(levels(df$player_id) == neymarId)
playerProbs = mcmcMat[,paste("probPlayer[", neymarLevel, "]", sep="")]
plotHDI(playerProbs, credMass=0.95, Title="Neymar HDI")
quantile(playerProbs)

plotInd(codaSamples, paste("probPlayer[", neymarLevel, "]", sep=""))

messiId = 154
messiLevel = which(levels(df$player_id) == messiId)
playerProbs = mcmcMat[,paste("probPlayer[", messiLevel, "]", sep="")]
plotHDI(playerProbs, credMass=0.95, Title="Messi HDI")
quantile(playerProbs)
