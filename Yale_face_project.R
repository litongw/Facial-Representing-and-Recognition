# Yale Face Project 
# set up working directory
setwd("~/R files/Yale_Face")
# include relevant libraries
library(pixmap) # data manipulate package
# create picture list and view list
pic_list<-1:38
view_list<-c('P00A+000E+00', 'P00A+005E+10', 'P00A+005E-10', 'P00A+010E+00')
# get directory structure
# load in all of the data by looping through the folders and storing the values in a list
dir_list_1 = dir(path="CroppedYale/",all.files=FALSE)
dir_list_2 = dir(path="CroppedYale/",all.files=FALSE,recursive=TRUE)
#==============================================================================
# Step 1. data manipulate
# 1.1 load the views of 4 lighting conditions for all subjects
# 1.2 convert each photo to a matrix and then to a vector
# 1.3 store the collection as a matrix where each row is a photo
pic_data = vector("list",length(pic_list)*length(view_list)) # prepare an empty list
faces_matrix <- vector()
# build outer loop
for (i in 1:length(pic_list)){
  this_face_row <- vector()
  # build inner loop
  for (j in 1:length(view_list)){
    this_filename = sprintf("CroppedYale/%s/%s_%s.pgm", dir_list_1[pic_list[i]] , dir_list_1[pic_list[i]] , view_list[j])
    this_face = read.pnm(file = this_filename)
    this_face_matrix = getChannels(this_face)
    this_face_matrix_row = as.vector(this_face_matrix)
    this_face_row = rbind( this_face_row , this_face_matrix_row )
  }
  faces_matrix = rbind(faces_matrix,this_face_row)
}
dim(faces_matrix) # size of matrix is 152*32256
#==============================================================================
# Step 2. compute main face
mean_face_vector<-colMeans(faces_matrix)
mean_face<-matrix(colMeans(faces_matrix),nrow=192,ncol=168,byrow=F)
mean_face_pix<-pixmapGrey(mean_face) # print as picture
plot(mean_face_pix)
#==============================================================================
# Step 3. principal components analysis of image matrix
# 3.1 subtract off the mean face
centered_faces<-apply(faces_matrix,1,'-',mean_face_vector)
# 3.2 pca
faces_pca<-prcomp(t(centered_faces))
# make a vector to store the variances captured by the components
n_comp <- length(faces_pca$x[,1])
pca_var <- mat.or.vec(n_comp,1)
for (i in 1:n_comp){
  if (i==1){
    pca_var[i] = faces_pca$sdev[i]^2
  }else{
    pca_var[i] = pca_var[i-1] + faces_pca$sdev[i]^2
  }
}

pca_var <- pca_var/pca_var[n_comp]*100
# cariances captured percentage plot
plot(pca_var,ylim=c(-2,102),xlab="Number of Components",ylab="Percentage of Variance Captured")
# add a line at 100 to show max level
abline(h=100,col="red")
#==============================================================================
# Step 4. "Eigenfaces"
# Each principal component is a picture, which are called “eigenfaces”
# explore first 9 eigenfaces
eigenface_matrix<-vector()
this_eigenface<-vector()
for (i in 1:9){
  eigenface_row<-faces_pca$rotation[,i]
  eigenface_row<-matrix(eigenface_row,nrow=192,ncol=168,byrow=F) #transform into matrix
  this_eigenface<-cbind(this_eigenface,eigenface_row)
  if ((i %% 3)==0){ #plot by 3x3
    eigenface_matrix<-rbind(eigenface_matrix,this_eigenface)
    this_eigenface<-vector()
  }
}
# plot eigen faces
plot(pixmapGrey(eigenface_matrix))
#==============================================================================
# Step 5. Use the eigenfaces to reconstruct yaleB05 P00A+010E+00.pgm
# 5.1 starting with the mean face, add in one eigenface at a time until reach 24 eigenfaces
# 5.2 starting with the mean face, add in five eigenfaces at a time until reach 120 eigenfaces.
# write a function to produce a 5 x 5 matrix of faces
reconst_matrix<-function(face_index,max_faces,by_faces,mean_face,faces_pca){
  face_by_eigen_matrix<-vector()
  face_by_eigen_row<-vector()
  face_by_eigen_vector<-mean_face
  face_temp<-face_by_eigen_vector
  face_by_eigen_row<-cbind(face_by_eigen_row,face_temp)
  # add eigenfaces
  for (i in 1:24){
    indice<- seq((i-1)*by_faces+1,i*by_faces,1) 
    eigenface_add<-mat.or.vec(32256,1)
    for (j in 1:length(indice)){
      ind_temp<-indice[j]
      eigenface_add<-eigenface_add + faces_pca$x[face_index,ind_temp]*faces_pca$rotation[,ind_temp]
    }
    face_by_eigen_vector<-face_by_eigen_vector + eigenface_add
    # transform back into matrix
    face_temp<-face_by_eigen_vector
    face_temp<-matrix(face_temp,nrow=192,ncol=168,byrow=F)
    face_by_eigen_row<-cbind(face_by_eigen_row,face_temp)
    if ((i %% 5) == 4){
      face_by_eigen_matrix<-rbind(face_by_eigen_matrix,face_by_eigen_row)
      face_by_eigen_row<-vector()
    }
  }
  return(face_by_eigen_matrix)
}

# 24 eigenfaces 
face_by_1<-reconst_matrix(20,24,1,mean_face,faces_pca)
plot(pixmapGrey(face_by_1))

# 120 eigenfaces
face_by_5<-reconst_matrix(20,120,5,mean_face,faces_pca)
plot(pixmapGrey(face_by_5))

#=============================================================================
# Step 6. facial recognition
library(FNN)
# 6.1 record the subject number and view of each row of face matrix in a data frame
pic_list_frame<-vector()
view_list_frame<-vector()
for(i in 1:length(pic_list)){
  this_pic_list_frame<-rbind(pic_list[i],pic_list[i],pic_list[i],pic_list[i])
  pic_list_frame<-rbind(pic_list_frame,this_pic_list_frame)
}
view_list_frame<-matrix(rep(view_list,38),ncol=1)

my_data_frame<-data.frame(pic_list_frame,view_list_frame)
# 6.2 divide face matrix into training set and test set
fm_size<-dim(faces_matrix)
# Use 4/5 of the data for training, 1/5 for testing
ntrain<-floor(fm_size[1]*4/5)
ntest<-fm_size[1]-ntrain
set.seed(1)
ind_train<-sample(1:fm_size[1],ntrain) #set of indices for the training data
ind_test<-c(1:fm_size[1])[-ind_train] #set of indices for the testing data
# find training and test data frame
train_frame<-my_data_frame[as.vector(ind_train),]
test_frame<-my_data_frame[as.vector(ind_test),]
# 6.3 do pca on training set
train_matrix<-faces_matrix[ind_train,]
mean_face_train<-colMeans(train_matrix)
mean_face_matrix<-matrix(mean_face_train,nrow=1)

train_centered<-scale(train_matrix,center=T,scale=F)
train_pca<-prcomp(train_centered)
# project testing data onto the first 25 loadings
loading_train<-train_pca$rotation[,1:25]
# test set data prepare
test_matrix<-faces_matrix[ind_test,]
test_centered<-t(apply(test_matrix,1,'-',mean_face_matrix))

test_projection<-test_centered%*%loading_train
train_projection<-train_centered%*%loading_train
# use 1NN to predict test data based on first 25PCs
nn_1<-knn(train_projection,test_projection,ind_train,k=1)
nn_1_subject<-my_data_frame[as.vector(nn_1),1]
# mis-classification rate
sum(nn_1_subject!= test_frame[,1]) # mis-classification rate = 0, good classification

#=============================================================================
# Step 7. Wider lighting condition ranges
view_list_new<-c("P00A-035E+15","P00A-050E+00","P00A+035E+15","P00A+050E+00")
pic_data = vector("list",length(pic_list)*length(view_list_new)) # prepare an empty list
faces_matrix <- vector()
# build outer loop
for (i in 1:length(pic_list)){
  this_face_row <- vector()
  # build inner loop
  for (j in 1:length(view_list_new)){
    this_filename = sprintf("CroppedYale/%s/%s_%s.pgm", dir_list_1[pic_list[i]] , dir_list_1[pic_list[i]] , view_list_new[j])
    this_face = read.pnm(file = this_filename)
    this_face_matrix = getChannels(this_face)
    this_face_matrix_row = as.vector(this_face_matrix)
    this_face_row = rbind( this_face_row , this_face_matrix_row )
  }
  faces_matrix = rbind(faces_matrix,this_face_row)
}
# record the subject number and view of each row of face matrix in a data frame
pic_list_frame<-vector()
view_list_new_frame<-vector()
for(i in 1:length(pic_list)){
  this_pic_list_frame<-rbind(pic_list[i],pic_list[i],pic_list[i],pic_list[i])
  pic_list_frame<-rbind(pic_list_frame,this_pic_list_frame)
}
view_list_new_frame<-matrix(rep(view_list_new,38),ncol=1)

my_data_frame<-data.frame(pic_list_frame,view_list_new_frame)
# divide face matrix into training set and test set
fm_size<-dim(faces_matrix)
# Use 4/5 of the data for training, 1/5 for testing
ntrain<-floor(fm_size[1]*4/5)
ntest<-fm_size[1]-ntrain
set.seed(2)
ind_train<-sample(1:fm_size[1],ntrain) #set of indices for the training data
ind_test<-c(1:fm_size[1])[-ind_train] #set of indices for the testing data
# find training and test data frame
train_frame<-my_data_frame[as.vector(ind_train),]
test_frame<-my_data_frame[as.vector(ind_test),]
# do pca on training set
train_matrix<-faces_matrix[ind_train,]
mean_face_train<-colMeans(train_matrix)
mean_face_matrix<-matrix(mean_face_train,nrow=1)

train_centered<-scale(train_matrix,center=T,scale=F)
train_pca<-prcomp(train_centered)
# project testing data onto the first 25 loadings
loading_train<-train_pca$rotation[,1:25]
# test set data prepare
test_matrix<-faces_matrix[ind_test,]
test_centered<-t(apply(test_matrix,1,'-',mean_face_matrix))

test_projection<-test_centered%*%loading_train
train_projection<-train_centered%*%loading_train
# use 1NN to predict test data based on first 25PCs
nn_1<-knn(train_projection,test_projection,ind_train,k=1)
nn_1_subject<-my_data_frame[as.vector(nn_1),1]
# mis-classification rate
sum(nn_1_subject!= test_frame[,1])
# result turned out 27 mis-classified subjects.
#=============================================================================
# Step 8. repeat Step 7 with 10 different seed
# try to prove high miss-classification rate is a normal situation under wider lighting condition
misclass<- NULL
for (i in 3:12) {
  fm_size<-dim(faces_matrix)
  ntrain<-floor(fm_size[1]*4/5)
  ntest<-fm_size[1]-ntrain
  set.seed(i)
  ind_train<-sample(1:fm_size[1],ntrain)
  ind_test<-c(1:fm_size[1])[-ind_train]
  train_frame<-my_data_frame[as.vector(ind_train),]
  test_frame<-my_data_frame[as.vector(ind_test),]
  train_matrix<-faces_matrix[ind_train,]
  test_matrix<-faces_matrix[ind_test,]
  mean_face_train<-colMeans(train_matrix)
  mean_face_matrix<-matrix(mean_face_train,nrow=1)
  train_centered<-t(apply(train_matrix,1,'-',mean_face_matrix))
  test_centered<-t(apply(test_matrix,1,'-',mean_face_matrix))
  test_projection<-test_centered%*%train_pca$rotation[,1:25]
  train_projection<-train_centered%*%train_pca$rotation[,1:25]
  nn_1<-knn(train_projection,test_projection,ind_train,k=1)
  nn_1_subject<-my_data_frame[as.vector(nn_1),1]
  misclass<- c(misclass, sum(nn_1_subject!=test_frame[,1]))
  cat(i , "done \n")
}
print(misclass)
# the numbers of misclassification are similar 
# this means dimension reduction by PCA using first 25 scores based on the wider lighting condition makes knn prediction very bad

