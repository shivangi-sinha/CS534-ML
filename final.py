import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def main():

    def import_data(filname):
        df = pd.read_csv(filname)
        return df

    def preprocessing(df,normalize = False,train = False,test = False):
    # Part a)
        del df['id']
    # Part b)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        del df['date']
    # Changing data types
        if train == True:
            df['waterfront'] = df['waterfront'].astype('category')
            df['condition'] = df['condition'].astype('category')
            df['grade'] = df['grade'].astype('category')
    
    # Making tables for numerical features
            name = []
            mean = []
            sd =[]
            ranges = []
            for column in df:
                i=[]
                m=[]
                s=[]
                r=[]
                if (df[column].name == 'dummy'):
                    continue
                if (df[column].dtype == np.int64 or df[column].dtype == np.float64):
            
                    i = df[column].name
                    m = df[column].mean()
                    s = df[column].std()
                    r = df[column].max() - df[column].min()
                    name.append(i)   
                    mean.append(m)
                    sd.append(s)
                    ranges.append(r)
            table = pd.DataFrame({'Name' : name,'Mean':mean,'Standard Deviation' :sd,'Range' : ranges})
            table.to_csv("table123.csv")
    # For categorical features 

            df2 = pd.DataFrame()    
            for column in df.select_dtypes(include=['category']):
                lab = len(df[column].unique())
                freq_v = list(((df[column].value_counts()/len(df[column]))*100)[0:lab])
                labels = df[column].unique()
            #df_v = df_v.sort_index()
                name = [df[column].name for i in range(0,lab)]
                df_v = pd.DataFrame({'Name': name,'Categories': labels,'Freq': freq_v})
                df2 = df2.append(df_v)
            df2.to_csv("cat123.csv")
        
            df['waterfront'] = df['waterfront'].astype('float64')
            df['condition'] = df['condition'].astype('float64')
            df['grade'] = df['grade'].astype('float64')
        
    # normalize
        def normal(data):
            cols = list(data.select_dtypes(include = 'number'))
            for i in range(0,len(cols)):
                col_range = data[cols[i]].max() - data[cols[i]].min()
                if col_range != 0:
                    data[cols[i]] = (data[cols[i]] - data[cols[i]].min()) / col_range
            return data
            
        if test == True:
            if normalize == True:
                df = normal(df)
        else:
            y = df.loc[:,['price']]
            x = df.drop(['price'],axis=1)
            if normalize == True:
                normal(x)
            return x,y    
        return df



    train= import_data("PA1_train.csv")
    validate = import_data("PA1_dev.csv")
    test = import_data("PA1_test.csv")
    train1 = train.copy()
    validate1 = validate.copy()
    X_train,Y_train= preprocessing(train,True,True,False)
    X_validation,Y_validation = preprocessing(validate,True,False,False)
    X_test = preprocessing(test,True,False,True)
    col = X_train.columns

    #Converting dataframe to matrix
    X_train= X_train.as_matrix().astype(float)
    Y_train = Y_train.as_matrix()
    X_validation= X_validation.as_matrix().astype(float)
    Y_validation = Y_validation.as_matrix()

    #cost function
    def L2cost_function(x,y,theta,lamda):
        n=y.size
        J=0
        #print(theta)
        h = np.matmul(x,theta)
        L2_cost = (lamda)*np.sum(np.matmul(theta[1:].T,theta[1:]))
        J= (1/2)*np.matmul(np.transpose(h-y),(h-y)) + L2_cost
        return J

    #Gradient calculation
    def gradient(x,y,theta,lamda):
        n = y.size
        h = np.matmul(x,theta)
        theta0 = theta[0,0]
        theta[0,0] = 0
        grad = np.matmul(x.T,(h-y)) + lamda*theta
        theta[0,0] = theta0
        return grad

    # Learning fucntion 

    def optimize(x,y,xval,yval,lamda,alpha,epsilon,limit=False):
        theta = np.zeros((x.shape[1],1))
        n_iter =0
        J = []
        Jval = []
        prev_grad = gradient(x,y,theta,lamda)
        cost = L2cost_function(x,y,theta,0)
        costval = L2cost_function(xval,yval,theta,0)
        J.append(cost)
        Jval.append(costval)
        n_iter += 1
        theta = theta-(alpha*prev_grad)
        grad = gradient(x,y,theta,lamda)
        test=0
        #print(grad)
        while (abs(np.linalg.norm(prev_grad) - np.linalg.norm(grad)) > epsilon):
            prev_grad = grad
            cost = L2cost_function(x,y,theta,0)
            costval = L2cost_function(xval,yval,theta,0)
            J.append(cost)
            Jval.append(costval)

            n_iter += 1
            theta = theta-(alpha*prev_grad)
            grad = gradient(x,y,theta,lamda)
            if cost > 10e100:
                print("explosion")
                test=1
                break
            if limit == False:
                if n_iter>100000:
                    print("max iterations exceeded")
                    break
            else:
                if n_iter>10000:
                    print("max iterations exceeded")
                    break
        return J,Jval,theta,n_iter,test
    
 ## Start of Part 1)    
    def plot_sse(sse1,lamda,alpha,n_iter):
        plt.plot(sse1, color='red', linestyle='dashed',linewidth = 1.5 )
        plt.title("SSE vs number of iteration at learning rate : {}".format(alpha))
        plt.xlabel("Number of iteration : {}".format(n_iter))
        plt.ylabel("SSE at learning rate :{}".format(alpha))


        plt.savefig("Part1"+str(alpha) + ".png", dpi = 300,bbox_inches="tight")
        plt.show()

    def sub_part1(X_train,Y_train,X_validation,Y_validation,lamda,alpha,epsilon):
        n = Y_train.size
        print("At learning rate:",alpha)
        writer = csv.writer(open("Part1training"+str(alpha)+".csv","w",newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
        sse1,sval,weights1,n_iter1,test = optimize(X_train,Y_train,X_validation,Y_validation,lamda,alpha,epsilon)
        w = weights1.size
        sse1 =np.concatenate(sse1).ravel().tolist()
        mse1 = L2cost_function(X_train,Y_train,weights1,lamda)/(n-w)
        writer.writerow(("SSE training =","learning rate",min(sse1),alpha))
        writer.writerow(("MSE training =","learning rate",mse1,alpha))
        if test==0:
            m = Y_validation.size
            s = np.matmul(X_validation,weights1) - Y_validation
            s = np.matmul(np.transpose(s),s)
            s =np.concatenate(s).ravel().tolist()
            con = min(sse1)
            mse2 = s[0]/(m-w)
            writer.writerow(["SSE validation =","learning rate",s,alpha])
            writer.writerow(["MSE validation =","learning rate",mse2,alpha])
        writer.writerow(["Weights","learning rate",weights1,alpha])
        plot_sse(sse1,lamda,alpha,n_iter1) 

    def part1():
        learning_rate = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
        lamda=0
        for i in learning_rate:
            sub_part1(X_train,Y_train,X_validation,Y_validation,lamda,i,0.05)
        print("Part 1 Finished")    
    ### Start of part 2)
    def part2(X_train,Y_train,X_validation,Y_validation):
        lambdas = [0,.001,.01,.1,1,10,100]
        SSEs_train = []
        SSEs_val = []
        weights = []
        iterations = []

        # training data
        for i in range(0,len(lambdas)):
            cost,costval,w,n,var = optimize(X_train,Y_train,X_validation,Y_validation,lambdas[i],1e-5,.05)
            SSEs_train.append(cost[-1])
            SSEs_val.append(costval[-1])
            weights.append(w)
            iterations.append(n)
        SSEs_train = np.concatenate(SSEs_train).ravel().tolist()
        SSEs_val = np.concatenate(SSEs_val).ravel().tolist()
        # Plotting
        plt.plot(np.log10(lambdas[1:]).tolist(), SSEs_train[1:], label = 'training')
        plt.plot(np.log10(lambdas[1:]).tolist(), SSEs_val[1:], label = 'validation', linestyle = 'dashed')
        plt.xlabel('log(lambda)')
        plt.ylabel('SSE')
        plt.legend(loc=2)
        plt.title('SSE vs. log(lambda)')
        plt.savefig('SSEvslambda.png', dpi=300)
        # Saving SSE 
        part2sse = pd.DataFrame([lambdas, SSEs_train, SSEs_val]).transpose()
        part2sse.columns = ['lambda','train','validation']
        part2sse.to_csv('part2sse.csv')
        weights0 = np.concatenate(weights[0]).ravel().tolist()
        weights1 = np.concatenate(weights[1]).ravel().tolist()
        weights2 = np.concatenate(weights[2]).ravel().tolist()
        weights3 = np.concatenate(weights[3]).ravel().tolist()
        weights4 = np.concatenate(weights[4]).ravel().tolist()
        weights5 = np.concatenate(weights[5]).ravel().tolist()
        weights6 = np.concatenate(weights[6]).ravel().tolist()
        part2weights = pd.DataFrame([weights0,weights1,weights2,weights3,weights4,weights5,weights6]).transpose()
        part2weights.columns = ['0','.001','.01','.1','1','10','100']
        part2weights['variable'] = col
        cols = part2weights.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        part2weights = part2weights[cols]
        part2weights.to_csv('part2weights.csv')
        print("Part 2 Finished")
        return weights

    ## Part 3)
    def part3(train,validation):
        Xtrain3,Ytrain3 = preprocessing(train,False,True)
        Xval3,Yval3 = preprocessing(validation)
        alphas = [1,0,1e-3,1e-6,1e-9,1e-15]
        SSEs_train3 = []
        SSEs_val3 = []
        weights3 = []
        iterations3 = []
        # training data
        for i in range(0,len(alphas)):
            cost,costval,w,n,t = optimize(Xtrain3,Ytrain3,Xval3,Yval3,0,alphas[i],.05,limit=True)
            SSEs_train3.append(cost)
            SSEs_val3.append(costval)
            weights3.append(w)
            iterations3.append(n)
         ## Plotting  
        for i in range(len(alphas)):
            if i != 1:
                plt.figure(i)
                plt.plot(range(iterations3[i]), np.concatenate(SSEs_train3[i]).ravel().tolist(), label = 'training')
                plt.plot(range(iterations3[i]), np.concatenate(SSEs_val3[i]).ravel().tolist(), label = 'validation', linestyle = 'dashed')
                plt.xlabel('iterations')
                plt.ylabel('SSE')
                plt.legend(loc=2)
                plt.title('SSE vs. iterations for learning rate = ' + str(alphas[i]))
                plt.savefig('part3alpha' + str(alphas[i]) + '.png', dpi=300)
            else:
                plt.figure(i)
                plt.plot(range(10), np.repeat(np.concatenate(SSEs_train3[i]).ravel().tolist(),10), label = 'training')
                plt.plot(range(10), np.repeat(np.concatenate(SSEs_val3[i]).ravel().tolist(),10),label = 'validation', linestyle = 'dashed')
                plt.xlabel('iterations')
                plt.ylabel('SSE')
                plt.legend(loc=2)
                plt.title('SSE vs. iterations for learning rate = ' + str(alphas[i]))
                plt.savefig('part3alpha' + str(alphas[i]) + '.png', dpi=300)
        print("Part 3 Finished")

    
    part1()
    weights =  part2(X_train,Y_train,X_validation,Y_validation)
    part3(train1,validate1)
    pred = np.dot(X_test,weights[0])
    np.savetxt("prediction.csv",pred)
main()
