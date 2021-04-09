import pandas as pd
import torch as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor


'''
•PassengerID（ID）
•Survived(存活与否)
•Pclass（客舱等级，较为重要）
•Name（姓名，可提取出更多信息）
•Sex（性别，较为重要）
•Age（年龄，较为重要）
•Parch（直系亲友）
•SibSp（旁系）
•Ticket（票编号）
•Fare（票价）
•Cabin（客舱编号）
•Embarked（上船的港口编号）
'''

def data_load(label_file_path,train_file_path,test_file_path):
    label=pd.read_csv(label_file_path)

    train=pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    PassengerId = test['PassengerId']
    all_data = pd.concat([train, test], ignore_index=True)

    #print(label)
    #print(train_data)
    #print(train.head())
    print(train.info())
    #print(train['Survived'].value_counts())


    #画柱状图
    
    
    sns.barplot(x="Sex", y="Survived",hue="Pclass", data=train) #性别
    plt.show()

    sns.barplot(x="Pclass", y="Survived", data=train) #乘客社会等级
    #plt.show()
    
    sns.barplot(x="SibSp", y="Survived", data=train) #配偶及兄弟姐妹数
    #plt.show()
    
    sns.barplot(x="Parch", y="Survived", data=train) #父母与子女数
    #plt.show()
    

    #散点密度图 
    facet = sns.FacetGrid(train, hue="Survived", aspect=2)
    facet.map(sns.kdeplot, 'Age', shade=True) #年龄与存活率
    facet.set(xlim=(0, train['Age'].max()))
    facet.add_legend()
    plt.xlabel('Age')
    plt.ylabel('density')
    #plt.show()



    #以bar的形式展示每个类别的数量
    sns.countplot('Embarked', hue='Survived', data=train)  #登港港口号
    #plt.show()
    sns.barplot(x="Embarked", y="Survived", data=train)
    #plt.show()
    
    
    


    #不同称呼的乘客幸存率不同
    all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    print(all_data['Name'])
    print(all_data['Title'] )
    Title_Dict = {}

    #合并一些无聊的称号
    Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    Title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))


    all_data['Title'] = all_data['Title'].map(Title_Dict)
    sns.barplot(x="Title", y="Survived", data=all_data)
    #plt.show()
    

    #FamilyLabel Feature(New)：家庭人数为2到4的乘客幸存率较高
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    sns.barplot(x="FamilySize", y="Survived", data=all_data)
    #plt.show()

    #按生存率把FamilySize分为三类，构成FamilyLabel特征。
    def Fam_label(s):
        if (s >= 2) & (s <= 4):
            return 2
        elif ((s > 4) & (s <= 7)) | (s == 1):
            return 1
        elif (s > 7):
            return 0

    all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_label)
    sns.barplot(x="FamilyLabel", y="Survived", data=all_data)
    #plt.show()
    
    #9)Deck Feature(New)：不同甲板的乘客幸存率不同
    all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
    all_data['Deck'] = all_data['Cabin'].str.get(0)
    sns.barplot(x="Deck", y="Survived", data=all_data)
    #plt.show()

    #10)TicketGroup Feature(New)：与2至4人共票号的乘客幸存率较高
    Ticket_Count = dict(all_data['Ticket'].value_counts())
    all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
    sns.barplot(x='TicketGroup', y='Survived', data=all_data)
    #plt.show()

    def Ticket_Label(s):
        if (s >= 2) & (s <= 4):
            return 2
        elif ((s > 4) & (s <= 8)) | (s == 1):
            return 1
        elif (s > 8):
            return 0

    all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
    sns.barplot(x='TicketGroup', y='Survived', data=all_data)
    #plt.show()


    #填充年龄
    age_df=all_data[['Age', 'Pclass','Sex','Title']]
    print(age_df)
    age_df = pd.get_dummies(age_df) # one_hot 编码
    print(age_df)
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    y=known_age[:,0]
    X=known_age[:,1:]
    rfr=RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=1)
    rfr.fit(X,y)
    predictedAges=rfr.predict(unknown_age[:,1:])
    #print(predictedAges)
    all_data.loc[(all_data.Age.isnull()),'Age']=predictedAges

    #填充embarked
    #print(all_data[all_data['Embarked'].isnull()])
    #print(all_data.groupby(by=["Pclass", "Embarked"]).Fare.median()) #Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C。
    all_data['Embarked'] = all_data['Embarked'].fillna('C')

    #print(all_data[all_data['Fare'].isnull()].values)
    fare = all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
    all_data['Fare'] = all_data['Fare'].fillna(fare)


    #同组识别
    all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())
    Surname_Count = dict(all_data['Surname'].value_counts())
    print(Surname_Count)
    all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: Surname_Count[x])
    Female_Child_Group = all_data.loc[
        (all_data['FamilyGroup'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
    Male_Adult_Group = all_data.loc[
        (all_data['FamilyGroup'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]

    Female_Child = pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
    Female_Child.columns = ['GroupCount']
    print("Female_Child",Female_Child)

    Male_Adult = pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
    Male_Adult.columns = ['GroupCount']
    print("Male_Adult",Male_Adult)

    # 因为普遍规律是女性和儿童幸存率高，成年男性幸存较低，
    # 所以我们把不符合普遍规律的反常组选出来单独处理。
    # 把女性和儿童组中幸存率为0的组设置为遇难组，把成年男性组中存活率为1的设置为幸存组，
    # 推测处于遇难组的女性和儿童幸存的可能性较低，处于幸存组的成年男性幸存的可能性较高。

    Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
    Dead_List = set(Female_Child_Group[Female_Child_Group.apply(lambda x: x == 0)].index)
    print(Dead_List)
    Male_Adult_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
    Survived_List = set(Male_Adult_List[Male_Adult_List.apply(lambda x: x == 1)].index)
    print(Survived_List)


    #为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
    train = all_data.loc[all_data['Survived'].notnull()]
    test = all_data.loc[all_data['Survived'].isnull()]

    train = all_data.loc[all_data['Survived'].notnull()]
    test = all_data.loc[all_data['Survived'].isnull()]
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Sex'] = 'male'
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Age'] = 60
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Title'] = 'Mr'
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Sex'] = 'female'
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Age'] = 5
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Title'] = 'Miss'

    print(all_data.isnull().sum()[all_data.isnull().sum() > 0]) #看看还有没有空值

    #todo 我总觉得不太科学，上面这块注释掉试试效果 试了，效果变差了，这么做的确好

    all_data = pd.concat([train, test])
    #print(all_data)
    all_data = all_data[
        ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilyLabel', 'Deck', 'TicketGroup']]  #去掉了passnger id

    #print(all_data)
    all_data = pd.get_dummies(all_data)
    train = all_data[all_data['Survived'].notnull()]
    test = all_data[all_data['Survived'].isnull()].drop('Survived', axis=1)
    X = train.values[:, 1:]
    y = train.values[:, 0]






    #4.建模和优化
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectKBest
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score


    '''  
    #随机森林版本
    pipe = Pipeline([('select', SelectKBest(k=20)),
                     ('classify', RandomForestClassifier(random_state=10, max_features='sqrt'))])

    param_test = {'classify__n_estimators': list(range(20, 50, 2)),
                  'classify__max_depth': list(range(3, 60, 3))}
    gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='roc_auc', cv=10)
    gsearch.fit(X, y)
    print(gsearch.best_params_, gsearch.best_score_)

    #2)训练模型

    select = SelectKBest(k=20)
    clf = RandomForestClassifier(random_state=10, warm_start=True,
                                 n_estimators=26,
                                 max_depth=6,
                                 max_features='sqrt')
    pipeline = make_pipeline(select, clf)   #https://blog.csdn.net/elma_tww/article/details/88427695
    pipeline.fit(X, y)

    cv_score = cross_val_score(pipeline, X, y, cv=10)
    print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

    predictions = pipeline.predict(test)
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
    submission.to_csv(r"./data/submission1.csv", index=False)
    '''
    #svm版本
    svc=make_pipeline(StandardScaler(),SVC(random_state=1))
    r=[0.001,0.01,0.1,1,10,50,100]
    PSVM=[{'svc__C':r,'svc__kernel':['linear']},
          {'svc__C': r, 'svc__gamma': r, 'svc__kernel': ['rbf']}
          ]
    GSSVM=GridSearchCV(estimator=svc,param_grid=PSVM,scoring='accuracy',cv=2)
    scores_svm=cross_val_score(GSSVM,X,y,scoring='accuracy',cv=5)
    model = GSSVM.fit(X, y)
    pred = model.predict(test)
    submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": pred.astype(np.int32)})
    submission.to_csv(r"./data/submission_svm.csv", index=False)


if __name__ == '__main__':
    label_file_path='./data/gender_submission.csv'
    train_file_path='./data/train.csv'
    test_file_path='./data/test.csv'
    data_load(label_file_path,train_file_path,test_file_path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
