import pandas as pd
import numpy as np
# 繪圖相關套件
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display
plt.style.use( 'ggplot' ) 
# 定義用來統計欄位缺漏值總數的函數
def Missing_Counts( Data ) : 
    missing = Data.isnull().sum()  # 計算欄位中缺漏值的數量 
    missing = missing[ missing>0 ]
    missing.sort_values( inplace=True ) 
    
    Missing_Count = pd.DataFrame( { 'ColumnName':missing.index, 'MissingCount':missing.values } )  # Convert Series to DataFrame
    Missing_Count[ 'Percentage(%)' ] = Missing_Count['MissingCount'].apply( lambda x:round(x/Data.shape[0]*100,2) )
    return  Missing_Count


df_train = pd.read_csv('../train.csv')
df_test = pd.read_csv('../test.csv')
submit = pd.read_csv('../gender_submission.csv')
print(df_train.shape)

df_data=df_train.append(df_test)

lr=0.0001




# print(df_train["Sex"])
# print(df_train["Pclass"])
# print(df_train["Embarked"])
# print(df_train["SibSp"])
# print(df_train["Parch"])
# exit()
df_train['Embarked'].fillna( 'S', inplace=True )
df_train['Fare'].fillna( df_train.Fare.median(), inplace=True )

#### 
#[0] sex
#[1] age
#[2] pclase
#[3] fare
#[4] family
#[5] Embarked
input_data=np.ones((891,7))
# input_data[:,0]=1.0
# print(input_data[0])
for i in range(891):
    if(df_train["Sex"][i]=="male"):
        input_data[i][0]=1
    else:
        input_data[i][0]=0
    if(df_train["Age"][i]>=16):
        input_data[i][1]=1
    else:
        input_data[i][1]=0
    
    input_data[i][2]=df_train["Pclass"][i]
    input_data[i][3]=df_train["Fare"][i]
    input_data[i][4]=df_train["SibSp"][i]+df_train["Parch"][i]
    # if(df_train[""])
    if(df_train["Embarked"][i]=="S"):
        input_data[i][5]=1
    elif(df_train["Embarked"][i]=="C"):
        input_data[i][5]=2
    elif(df_train["Embarked"][i]=="Q"):
        input_data[i][5]=3

    # print(input_data[i])
# weight=np.random.rand(7)
# output=np.zeros((891,1),dtype=float)

###### how to implement aggregation perceptrons?



def tanh_differential(x):
    tanh_diff=1-(np.tanh(x)*np.tanh(x))

    return tanh_diff
def forward(s3,w3,x2,s2,w2,x1,s1,w1,x0):
    # x0=input
    # w1=np.random.randn(3,7)
    # #s1=x0.T.dot(w1)
    # s1=np.zeros(3)
    # x1=np.ones(4)
    # w2=np.random.randn(4,3)
    # #s2=x1.T.dot(w2)
    # s2=np.zeros(3)
    # x2=np.ones(4)
    # w3=np.random.randn(4)
    # #s3=x2.T.dot(w3)
    # s3=0.0
    for i in range(w1.shape[0]):
        s1[i]=x0.T.dot(w1[i])
        x1[i]=np.tanh(s1[i])
    for i in range(w2.shape[0]):
        s2[i]=x1.T.dot(w2[i])
        x2[i]=np.tanh(s2[i])
    s3=x2.T.dot(w3)
    return s3,w3,x2,s2,w2,x1,s1,w1

def backward():
    return 0
def backward_1L(x0,w1,s1,w2,delta_2):
    delta_1=np.zeros(s1.shape[0])

    """ compute delta """

    for j in range(s1.shape[0]):
        sum=0
        for k in range(w2.shape[0]):
            sum=sum+delta_2[k]*w2[k][j]*tanh_differential(s1[j])
        delta_1[j]=sum

    # for j in range(w1.shape[0]):
    #     sum=0
    #     for k in range(delta_2.shape[0]):
    #         sum=sum+delta_2[k]*w2[k][j]*tanh_differential(s1[j])
    #     delta_1[j]=sum
    
    """ update weight """
    # for k in range(delta_1.shape[0]):
    #     for j in range(delta_1.shape[1]):
    #         w1[]
    for j in range(s1.shape[0]):
        for i in range(x0.shape[0]):
            w1[j][i]=w1[j][i]-lr*delta_1[j]*x0[i]
    # for j in range(s2.shape[0]):
    #     for i in range(x1.shape[0]):
    #         w1[j][i]=w1[j][i]-lr*delta_1[j]*x0[i]


    return delta_1,w1
#--------------------------------------------- may be this is faliure
# need check
# check done

def backward_2L(x1,w2,s2,w3,delta_3):
    delta_2=np.zeros(s2.shape[0])
    
    """ compute delta """
    for j in range(s2.shape[0]):
        delta_2[j]=delta_3*w3[j]*tanh_differential(s2[j])

    """ update weight """    
    for j in range(s2.shape[0]):
        for i in range(x1.shape[0]):
            w2[j][i]=w2[j][i]-lr*delta_2[j]*x1[i]


    return delta_2,w2


def backward_3L(x2,w3,s3,y):
    """ differentiable L_w each element """

    delta_3=0.0

    for i in range(w3.shape[0]):
        delta_3=(-2)*(y-s3)
        """ partial differential """
        w3[i]=w3[i]-lr*delta_3*x2[i]
    return delta_3,w3
def main():
    # random.seed(10)
    idx=0

    """ initial value """
    x0=input_data[idx]
    # w1=np.random.randn(6,7)
    # #s1=x0.t.dot(w0)
    # s1=np.zeros(7)
    # # x1=tanh(s1)
    # x1=np.zeros(7)
    # w2=np.random.randn(6,7)
    # #s2=x1.t.dot(w1)
    # s2=np.zeros(7)
    # #x2=tanh(s2)
    # x2=np.zeros(7)
    # w3=np.random.randn(7)

    w1=np.random.randn(3,7)
    #s1=x0.T.dot(w1[i])
    s1=np.zeros(3)
    x1=np.ones(4)
    w2=np.random.randn(3,4)
    #s2=x1.T.dot(w2[i])
    s2=np.zeros(3)
    x2=np.ones(4)
    w3=np.random.randn(4)
    #s3=x2.T.dot(w3)
    s3=0.0


    
    #s3=x2.t.dot(w2)
    #s3=ans
    s3=0.0
    for epoch in range(5000):
        # idx=1
        accurate=0.0
        correct=0
        loss=0.0

        for idx in range(input_data.shape[0]):
            if(df_train["Survived"][idx]==1):
                y=1
            else:
                y=-1
            x0=input_data[idx]
            s3,w3,x2,s2,w2,x1,s1,w1=forward(s3,w3,x2,s2,w2,x1,s1,w1,x0)
            # print("idx:\t"+str(idx)+"\t\tloss:\t"+'%.6f' % (y-s3)**2)
            loss+=(y-s3)**2
            # print("y:\t"+'%.6f' % y)
            # print("s3:\t"+'%.6f' % s3)
            delta_3,w3=backward_3L(x2,w3,s3,y)
            delta_2,w2=backward_2L(x1,w2,s2,w3,delta_3)
            delta_1,w1=backward_1L(x0,w1,s1,w2,delta_2)
            if(y==1):
                if(s3>=0):
                    correct+=1
                else:
                    pass
            if(y==-1):
                if(s3>=0):
                    pass
                else:
                    correct+=1
        loss=loss/input_data.shape[0]
        accurate=correct/input_data.shape[0]
        print("epoch:\t"+str(epoch)+"\t"+str(correct)+"/"+str(input_data.shape[0])+"\tacc:\t\t"+str(accurate)+"\tloss:\t\t"+str(loss))
        """ test process """
        if(correct>=720):
            test=np.ones((df_test.shape[0],7))
            answer=np.zeros((df_test.shape[0],2),dtype=int)
            for i in range(df_test.shape[0]):
                if(df_test["Sex"][i]=="male"):
                    test[i][0]=1
                else:
                    test[i][0]=0
                if(df_test["Age"][i]>=16):
                    test[i][1]=1
                else:
                    test[i][1]=0
                
                test[i][2]=df_test["Pclass"][i]
                test[i][3]=df_test["Fare"][i]
                test[i][4]=df_test["SibSp"][i]+df_test["Parch"][i]
                # if(df_train[""])
                if(df_test["Embarked"][i]=="S"):
                    test[i][5]=1
                elif(df_test["Embarked"][i]=="C"):
                    test[i][5]=2
                elif(df_test["Embarked"][i]=="Q"):
                    test[i][5]=3


                x0=test[i]
                s3,w3,x2,s2,w2,x1,s1,w1=forward(s3,w3,x2,s2,w2,x1,s1,w1,x0)
                if(s3>=0):
                    survived=1
                else:
                    survived=0
                answer[i][1]=survived
            answer[:,0]=df_test["PassengerId"]
            answer_csv={"PassengerId":answer[:,0],"Survived":answer[:,1]}
            answer_csv_df = pd.DataFrame(answer_csv)
            answer_csv_df.to_csv("kaggel_answer.csv",index=False)    
            break
        # print()
        # print()
        # print()
        # s3,w3,x2,s2,w2,x1,s1,w1=forward(s3,w3,x2,s2,w2,x1,s1,w1,x0)
    
    # if(y==1):
    #     print("ans:\tsurvived")
    # else:
    #     print("ans:\tdead")
    # if(s3>0):
    #     print("pre:\tsurvive")
    # else:
    #     print("pre:\tdead")
    # print(output)
    # print("loss:\t"+str((y-s3)**2))
    #D
    # """ mse """
    # print("loss:\t"+'%.6f' % (y-s3)**2)
    # exit()
if __name__ == '__main__':
    main()


