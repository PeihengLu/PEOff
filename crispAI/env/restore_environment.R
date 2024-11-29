# restore_environment.R

# Read the CSV file into a data frame
package_info <- read.csv("R_env.csv", stringsAsFactors = FALSE)

# Install packages from the data frame
for (row in 1:nrow(package_info)) {
  install.packages(
    package_info[row, "Package"], 
    version = package_info[row, "Version"],
    dependencies = TRUE,
    # use the Tsinghua mirror while in China
    # original us mirror: 
    # repos = "https://cran.r-project.org"
    repos = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"
  )
}

BiocManager::install("BiocVersion")
BiocManager::install("BiocStyle")
BiocManager::install("NuPoP")

# Load installed packages
lapply(package_info$Package, library, character.only = TRUE)