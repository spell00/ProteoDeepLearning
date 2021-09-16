library(MALDIquant)
library(MALDIquantForeign)
library(MSnbase)


# path <- 'D://workbench//data//spectro//Canis//positive_mzXML'
path <- '../../inputs_300'
out_filename <- "data/output"
# path <- '/mnt/d/workbench/data/spectro/Canis/positive_mzXML'
files <- list.files(path, full.names=TRUE)
# rawdata <- readMSData(files, mode="onMemory")
# spectra <- import(files, verbose=FALSE)
# x <- readMSData(files, centroided. = c(FALSE, TRUE, FALSE), mode = "onDisk")
spectra <- readMSData(files, mode='onDisk')
tics <- tic(spectra)
tics <- tics[tics > 1e4]

intensities <- intensity(bin(spectra, binSize=0.1))

spectra <- transformIntensity(spectra, method = "sqrt")
# Baseline Correction
spectra <- removeBaseline(spectra, method = "SNIP", iterations=100)
# Intensity Calibration / Normalization
spectra <- calibrateIntensity(spectra, method="TIC")
# Spectra alignment
spectra <- alignSpectra(spectra, halfWindowSize = 5, allowNoMatches= TRUE,  tolerance = 0.002, warpingMethod="cubic")
# Peaks detection
peaks <- detectPeaks(spectra, method = "MAD", halfWindowSize = 5, SNR = 3)
# Create an intensity matrix
intensity_matrix <- intensityMatrix(peaks, spectra)

# write the output file
file_sp <- file(out_filename, open="wt")
write.table(intensity_matrix, file_sp, sep=" ", row.names=FALSE)
