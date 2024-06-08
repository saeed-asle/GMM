import numpy as np
import matplotlib.pyplot as plt
np.random.seed(40)
 
# Group 1 parameters
group1_mean=np.array([-1,-1])
group1_cov=np.array([[0.8,0], [0,0.8]])

# Group 2 parameters
group2_mean=np.array([1,1])
group2_cov=np.array([[0.75,-0.2], [-0.2,0.6]])

# Generate data
group1=np.random.multivariate_normal(group1_mean,group1_cov,700)
group2=np.random.multivariate_normal(group2_mean,group2_cov,300)

# Combine the data from both groups
data=np.concatenate((group1,group2))

plt.scatter(group1[:,0],group1[:,1],label='Group 1')
plt.scatter(group2[:,0],group2[:,1],label='Group 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data')
plt.legend()
plt.show()

plt.scatter(data[:,0],data[:,1],label='data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Data')
plt.legend()
plt.show()
"""
expectat_part(data, means, cov, weights):
For each data point x_i in data:
  For each component k:
    Compute the probability density function (PDF) of x_i belonging to component k:
      PDF(x_i, k) = (1 / sqrt((2 * pi)^D * det(cov_k))) * exp(-0.5 * (x_i - means_k)^T * inv(cov_k) * (x_i - means_k))
    Calculate the responsibility of component k for data point x_i:
      result[i, k] = weights[k] * PDF(x_i, k) / sum(weights[j] * PDF(x_i, j) for all j)
Return the result matrix of responsibilities
"""
def expectat_part(data,means,cov,weights):
    result=np.zeros((data.shape[0],len(weights)))
    for j in range(len(weights)):
        cov_reg=cov[j]+np.eye(cov.shape[1])*1e-6
        exp=np.exp(-0.5*np.sum(np.dot((data-means[j]),np.linalg.inv(cov_reg))*(data-means[j]), axis=1))
        den=np.sqrt((2*np.pi)**data.shape[1]*np.linalg.det(cov_reg))
        result[:,j]=weights[j]*exp/den
    result/=np.max(result,axis=1,keepdims=True)  # Scale result
    return result

"""
maximize_part(data,result):
For each component k:
  Compute the updated weight of component k:
    weights[k] = sum(result[:, k]) / N
  Compute the updated mean of component k:
    means[k] = sum(result[i, k] * x_i for all data points x_i) / sum(result[:, k])
  Compute the updated covariance matrix of component k:
    cov[k] = sum(result[i, k] * (x_i - means[k])^T * (x_i - means[k]) for all data points x_i) / sum(result[:, k])
Return the updated means, covariances, and weights

"""
def maximize_part(data,result):
    means=np.zeros((result.shape[1],data.shape[1]))
    cov=np.zeros((result.shape[1],data.shape[1],data.shape[1]))
    w=np.zeros(result.shape[1])
    for j in range(result.shape[1]):
        w[j]=np.mean(result[:,j])
        means[j]=np.dot(result[:,j],data)/np.sum(result[:,j])
        cov[j]=np.dot(result[:,j]*(data-means[j]).T,(data-means[j]))/np.sum(result[:,j])
        cov[j]+=np.eye(data.shape[1])*1e-6  # Add a small regularization term
    return means,cov,w
num_com=2
num_iter=1000
littile=1e-6
# Initialize means randomly
means=np.random.uniform(low=data.min(axis=0),high=data.max(axis=0),size=(num_com,data.shape[1]))

cov=np.array([group1_cov,group2_cov])
w=np.ones(num_com)/num_com

res_log=0
num_res_iter=0
for iter in range(num_iter):
    result=expectat_part(data, means, cov, w)
    means,cov,w=maximize_part(data, result)
    log_new=np.sum(np.log(np.sum(result, axis=1)))
    if np.abs(log_new-res_log)<littile:
        num_res_iter=iter
        break
    res_log=log_new

predicted_result=np.argmax(result, axis=1)
predicted_group1=data[predicted_result==0]
predicted_group2=data[predicted_result==1]
print("means: \n{}".format(means))
print("covariances: \n{}".format(cov))
print("weights: \n{}".format(w))

plt.scatter(predicted_group1[:,0],predicted_group1[:,1],label='Predicted Group 1')
plt.scatter(predicted_group2[:,0],predicted_group2[:,1],label='Predicted Group 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Predicted Groups \nConverged at iteration: {}".format(num_res_iter))
plt.legend()
plt.show()
