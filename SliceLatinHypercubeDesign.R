library(SLHD)
# < z4x2 > ===================================================================
## < size: 5 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=4, # number of slices
    m=5, # number of design points in each slice
    k=2, # number of factors
    nstarts=100 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z4x2/size5/",i,".txt"),
    row.names=F,
    col.names=F
  )
}
## < size: 4 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=4, # number of slices
    m=4, # number of design points in each slice
    k=2, # number of factors
    nstarts=100 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z4x2/size4/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 3 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=4, # number of slices
    m=3, # number of design points in each slice
    k=2, # number of factors
    nstarts=100 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z4x2/size3/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 2 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=4, # number of slices
    m=2, # number of design points in each slice
    k=2, # number of factors
    nstarts=100 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z4x2/size2/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

# < z2x2 > ===================================================================
## < size: 64 > ==============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=64, # number of design points in each slice
    k=2, # number of factors
    nstarts=5 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size64/",i,".txt"),
    row.names=F,
    col.names=F
  )
}
## < size: 32 > ==============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=32, # number of design points in each slice
    k=2, # number of factors
    nstarts=50 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size32/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 16 > ==============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=16, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size16/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 10 > ==============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=10, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size10/",i,".txt"),
    row.names=F,
    col.names=F
  )
}
## < size: 8 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=8, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size8/",i,".txt"),
    row.names=F,
    col.names=F
  )
}
## < size: 6 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=6, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size6/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 5 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=5, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size5/",i,".txt"),
    row.names=F,
    col.names=F
  )
}
## < size: 4 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=4, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size4/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 3 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=3, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size3/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 2 > ===============================================================
for (i in 1:100){
  
  sample_design = maximinSLHD(
    t=2, # number of slices
    m=2, # number of design points in each slice
    k=2, # number of factors
    nstarts=1 # number of random starts
  )
  
  write.table(
    x=sample_design$StandDesign,
    file=paste0("SLHD_initial_data/z2x2/size2/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

# < z5x1 > ===================================================================
## < size: 4 > ===============================================================
size_each_slice=4
for (i in 1:100){
  # qualitative factor
  z = sample(rep(c(1:5), each=size_each_slice), size=5*size_each_slice, replace=FALSE)
  # quantitative factor
  x = round(lhs::maximinLHS(n=size_each_slice*5, k=1) / 0.005) * 0.005
  
  
  write.table(
    x=cbind(z, x),
    file=paste0("SLHD_initial_data/z5x1/size4/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 3 > ===============================================================
size_each_slice=3
for (i in 1:100){
  # qualitative factor
  z = sample(rep(c(1:5), each=size_each_slice), size=5*size_each_slice, replace=FALSE)
  # quantitative factor
  x = round(lhs::maximinLHS(n=size_each_slice*5, k=1) / 0.005) * 0.005
  
  
  write.table(
    x=cbind(z, x),
    file=paste0("SLHD_initial_data/z5x1/size3/",i,".txt"),
    row.names=F,
    col.names=F
  )
}

## < size: 2 > ===============================================================
size_each_slice=2
for (i in 1:100){
  # qualitative factor
  z = sample(rep(c(1:5), each=size_each_slice), size=5*size_each_slice, replace=FALSE)
  # quantitative factor
  x = round(lhs::maximinLHS(n=size_each_slice*5, k=1) / 0.005) * 0.005
  
  
  write.table(
    x=cbind(z, x),
    file=paste0("SLHD_initial_data/z5x1/size2/",i,".txt"),
    row.names=F,
    col.names=F
  )
}