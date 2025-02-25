# -*- coding: utf-8 -*-

# TsFresh时间序列的特征工程，包括特征生成和特征选择
import os
import gc
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from numpy import vstack, array, nan
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from itertools import combinations
# from tsfresh import extract_features
# from tsfresh import select_features
# from tsfresh.utilities.dataframe_functions import impute

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer
from sklearn.impute import SimpleImputer  # 处理缺失值的类
from sklearn.inspection import permutation_importance  # 用于计算特征的排列重要性
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# import eli5  # 解释机器学习模型和预测结果的库
# from eli5.sklearn import PermutationImportance

import warnings

warnings.filterwarnings('ignore')

# Mac OS
plt.rcParams['font.sans-serif'] = 'Songti Sc'
plt.rcParams['axes.unicode_minus'] = False

# 设置页面标题
st.title('特征工程实现流程')


# 创建存储文件夹的函数
def create_folder(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


# 箱型图过滤函数
def box_filters(df):
    for col in df.columns:
        # 时间列和设备列不过滤
        # exclusive_column=df.columns.str.contains('时间|设备', case=False)
        date_col = df.filter(like='时间').columns[0]
        device_col = df.filter(like='设备').columns[0]
        if col not in [date_col, device_col]:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
    return df[(df >= lower_bound) & (df <= upper_bound)].dropna()


# 定义一个函数来进行箱型图过滤
def box_filter(df, columns):
    filtered_df = df.copy()
    for col in columns:
        Q1 = filtered_df[col].quantile(0.25)
        Q3 = filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df.dropna()


# 3-sigma过滤函数
def sigma_filter(df, columns):
    filtered_df = df.copy()
    for col in columns:
        # 时间列和设备列不过滤
        mean = filtered_df[col].mean()
        std_dev = filtered_df[col].std()
        lower_bound = mean - 3 * std_dev
        upper_bound = mean + 3 * std_dev
        filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    return filtered_df.dropna()


# 移动平均过滤函数
def moving_average_filter(df, columns, window=5):
    filtered_df = df.copy()
    for col in columns:
        filtered_df[col + f"move_average"] = filtered_df[col].rolling(window=window).mean()
    return filtered_df.dropna()


# 选择过滤方式
def filter_data(df, method):
    # 进行箱型图过滤
    date_col = df.filter(like='时间').columns[0]
    device_col = df.filter(like='设备').columns[0]
    inclusive_column = df.iloc[:, ~df.columns.str.contains('时间|设备', case=False)].columns
    if method == '箱型图过滤':
        return box_filter(df, inclusive_column)
    elif method == '3-Sigma过滤':
        return sigma_filter(df, inclusive_column)
    elif method == '移动平均过滤':
        return moving_average_filter(df, inclusive_column)
    else:
        return df


def create_sidebar():
    st.sidebar.title("特征工程6大步骤")
    # st.sidebar.markdown("1. 上传数据")
    # st.sidebar.markdown("2. 数据可视化")
    # st.sidebar.markdown("3. 数据清洗")
    # st.sidebar.markdown("4. 工况分割")
    # st.sidebar.markdown("5. 特征生成")
    # st.sidebar.markdown("6. 特征选择")
    # with st.sidebar.expander("小步骤列表"):
    #     st.sidebar.markdown("- 小步骤6.1：小步骤6.1的描述")
    #     st.sidebar.markdown("- 小步骤6.2：小步骤6.2的描述")

    # toggle_state = st.toggle('点击展开/收起')
    # if toggle_state:
    #     st.write('这是展开的内容')

    # 侧边栏
    with st.sidebar:
        # 检查session_state中是否已初始化步骤状态，若没有则进行初始化
        if 'step_states' not in st.session_state:
            st.session_state.step_states = {f'step{i}': False for i in range(1, 7)}

        if st.button("1、输入数据路径", key="step1_button"):
            st.session_state.step_states['step1'] = not st.session_state.step_states['step1']

        if st.button("2、数据可视化", key="step2_button"):
            st.session_state.step_states['step2'] = not st.session_state.step_states['step2']

        if st.session_state.step_states['step2']:
            st.markdown("- 2.1、趋势图")
            st.markdown("- 2.2、散点图")
            st.markdown("- 2.3、直方图")
            st.markdown("- 2.4、箱型图")

        if st.button("3、数据清洗", key="step3_button"):
            st.session_state.step_states['step3'] = not st.session_state.step_states['step3']

        if st.session_state.step_states['step3']:
            st.markdown("- 3.1、箱型图过滤")
            st.markdown("- 3.2、3-sigma过滤")
            st.markdown("- 3.3、移动平均过滤")

        if st.button("4、工况分割", key="step4_button"):
            st.session_state.step_states['step4'] = not st.session_state.step_states['step4']

        if st.session_state.step_states['step4']:
            pass

        if st.button("5、特征生成", key="step5_button"):
            st.session_state.step_states['step5'] = not st.session_state.step_states['step5']

        if st.session_state.step_states['step5']:
            st.markdown("- 5.1、tsfresh特征生成")
            st.markdown("- 5.2：ECSM(Exceedance Combination Selection Model)特征生成")

        # 步骤1
        if st.button("6、特征选择", key="step6_button"):
            st.session_state.step_states['step6'] = not st.session_state.step_states['step6']

        if st.session_state.step_states['step6']:
            st.markdown("- 6.1：filter-方差选择法")
            st.markdown("- 6.2：filter-卡方检验")
            st.markdown("- 6.3：filter-相关系数法")
            st.markdown("- 6.4：wrapper-RFE递归特征消除法")
            st.markdown("- 6.5：embedded-基于L1惩罚项的特征选择法")
            st.markdown("- 6.6：embedded-结合L1和L2惩罚项的特征选择法")
            st.markdown("- 6.7：embedded-基于树模型的特征选择法")
            st.markdown("- 6.8：结合SVM和L1惩罚项的特征选择")
            st.markdown("- 6.9：LASSO")


def feature_selection_variance():
    """
    特征选择-方差选择法
    :return:
    """
    ## 特征选择filter-方差选择法,参数threshold为方差的阈值
    std_selector = VarianceThreshold(threshold=0.1)
    std_select_array = std_selector.fit_transform(df_noNA)
    # print(selector.variances_) #每个特征值的标准差
    # print(selector.get_support(indices=True)) #选择的特征值的index

    # 特征选择filter-方差选择法，将选择后的特征转换为DataFrame
    X_selected_filter_std = pd.DataFrame(std_select_array,
                                         columns=df_copy.columns[std_selector.get_support(indices=True)])

    # 特征选择filter-方差选择法，计算选择后的特征的标准差
    output_X_selected_std = np.std(X_selected_filter_std, axis=0)
    output_std_df = output_X_selected_std.to_frame()
    output_std_df['特征名称'] = output_std_df.index
    output_std_df.columns = ['标准差', '特征名称']

    # 特征选择filter-方差选择法，选择的特征和衡量特征的指标保存为文件
    X_selected_filter_std.to_excel(os.path.join(result_folder,subfolders['filter_variance_selection_folder'],'选择的特征-filter-方差选择法.xlsx'))
    output_std_df.to_excel(os.path.join(result_folder,subfolders['filter_variance_selection_folder'],'选择的特征的衡量指标-filter-方差选择法.xlsx'))
    st.success("方差选择法已完成!")


def feature_selection_chi2_test():
    """
    特征选择-卡方检验
    # 特征选择filter-卡方检验，适用于离散变量，要求数值为正值，使用区间缩放后且填充过缺失值的数据
    # 特征选择filter-卡方检验，选择K个最好的特征，返回选择特征后的数据
    :return:
    """
    # 特征选择filter-卡方检验，划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df_MinMax, y, test_size=0.2, random_state=0)

    # 特征选择filter-卡方检验，使用卡方选择，选择单变量特征选择K个最佳特征
    chi2(X_train, y_train)  # 查看卡方的两个指标，一是各个特征变量的卡方统计量值，二是各个特征变量的P统计量值，
    k = 10  # 要选择的特征数
    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector_array = chi2_selector.fit_transform(X_train, y_train)

    # 特征选择filter-卡方检验，将选择后的特征转换为DataFrame
    X_selected_filter_chi = pd.DataFrame(chi2_selector_array,
                                         columns=df_copy.columns[chi2_selector.get_support(indices=True)])

    # 特征选择filter-卡方检验，输出选择的特征、对应的卡方值和P-value
    output_chi_df = pd.DataFrame({'Feature': df_copy.columns[chi2_selector.get_support(indices=True)],
                                  'Chi-Square': chi2_selector.scores_[chi2_selector.get_support(indices=True)],
                                  'P-Value': chi2_selector.pvalues_[chi2_selector.get_support(indices=True)]})

    # 特征选择filter-卡方检验，选择的特征和衡量特征的指标保存为文件
    X_selected_filter_chi.to_excel(os.path.join(result_folder,subfolders['filter_chi2_estimator_folder'],'选择的特征-filter-卡方检验.xlsx'))
    output_chi_df.to_excel(os.path.join(result_folder,subfolders['filter_chi2_estimator_folder'],'选择的特征的衡量指标-filter-卡方检验.xlsx'))
    st.success("卡方检验法已完成!")

def feature_selection_correlation_coefficient():
    """
    特征选择filter-相关系数法
    特征选择filter-相关系数法，初始化一个字典来存储相关系数和p值
    :return:
    """
    correlation_dict = {}
    features = df_copy.columns

    # 特征选择filter-相关系数法，计算相关系数
    for feature in df_copy.columns:
        corr, p_value = pearsonr(df_copy[feature], y)
        correlation_dict[feature] = (corr, p_value)

    # 特征选择filter-相关系数法，将相关系数转换为DataFrame以便查看
    correlation_df = pd.DataFrame.from_dict(correlation_dict, orient='index', columns=['Correlation', 'P-value'])

    # 特征选择filter-相关系数法，设置相关系数选择的阈值
    threshold = 0.5

    # 特征选择filter-相关系数法，选择相关性强的特征，输出选择的特征的相关系数和特征表
    output_correlation_df = correlation_df[correlation_df['Correlation'].abs() > threshold]
    selected_features = correlation_df[correlation_df['Correlation'].abs() > threshold].index.tolist()
    X_selected_filter_correlation = df_copy[selected_features]

    # 特征选择filter-相关系数法，选择的特征和衡量特征的指标保存为文件
    X_selected_filter_correlation.to_excel(os.path.join(result_folder,subfolders['filter_correlation_coefficient_folder'],'选择的特征-filter-相关系数法.xlsx'))
    output_correlation_df.to_excel(os.path.join(result_folder,subfolders['filter_correlation_coefficient_folder'],'选择的特征的衡量指标-filter-相关系数法.xlsx'))
    st.success("相关系数法已完成!")

def feature_selection_RFE():
    """
    特征选择wrapper-RFE递归特征消除法
    :return:
    """
    # 特征选择wrapper-RFE，创建一个用于RFE选择特征的分类模型
    rfe_selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)

    ## 特征选择wrapper-RFE，模型拟合RFE
    rfe_selector.fit(df_noNA, y)
    rfe_selector_array = rfe_selector.fit_transform(df_noNA, y)

    # 特征选择wrapper-RFE，将选择后的特征转换为DataFrame
    X_selected_wrapper_rfe = pd.DataFrame(rfe_selector_array,
                                          columns=df_copy.columns[rfe_selector.get_support(indices=True)])

    ## 特征选择wrapper-RFE，保存选择的特征和ranking
    record_result_all = []
    # print('Selected Features:')
    for i in range(len(df_copy.columns)):
        output_wrapper_rfe = pd.DataFrame()
        if rfe_selector.support_[i]:
            #        print(df_copy.columns[i])
            output_wrapper_rfe['feature'] = [df_copy.columns[i]]
            output_wrapper_rfe['ranking'] = [rfe_selector.ranking_[i]]
            record_result_all.append(output_wrapper_rfe)
    output_wrapper_rfe_selected = pd.concat(record_result_all)

    # 特征选择wrapper-RFE，获取全部特征的ranking
    features = df_copy.columns
    rfe_feature_ranking = rfe_selector.ranking_
    # 特征选择wrapper-RFE，创建全部特征ranking的DataFrame
    output_wrapper_rfe_all = pd.DataFrame({'Feature': features, 'Ranking': rfe_feature_ranking})
    output_wrapper_rfe_all = output_wrapper_rfe_all.sort_values(by='Ranking')

    # 特征选择wrapper-RFE，选择的特征和衡量特征的指标保存为文件
    X_selected_wrapper_rfe.to_excel(os.path.join(result_folder,subfolders['filter_RFE_folder'],'选择的特征-wrapper-RFE.xlsx'))
    output_wrapper_rfe_selected.to_excel(os.path.join(result_folder,subfolders['filter_RFE_folder'],'选择的特征的衡量指标-wrapper-RFE.xlsx'))
    output_wrapper_rfe_all.to_excel(os.path.join(result_folder,subfolders['filter_RFE_folder'],'全部特征的衡量指标-wrapper-RFE.xlsx'))

    # 特征选择wrapper-RFE，选择的特征的ranking可视化
    plt.figure(figsize=(10, 15))
    sns.barplot(x='feature', y='ranking', data=output_wrapper_rfe_selected)
    plt.xticks(rotation=90)
    plt.title('Feature Ranking')
    plt.xlabel('Ranking')
    plt.ylabel('Feature')
    plt.savefig(os.path.join(result_folder,subfolders['filter_RFE_folder'],'特征可视化-wrapper-RFE.jpg'))
    st.success("RFE方法已完成!")

def feature_selection_embedded_based_on_L1():
    """
    基于L1惩罚项的特征选择法
    :return:
    """
    # 特征选择embedded-基于惩罚项的特征选择法，带L1惩罚项的逻辑回归作为基模型的特征选择
    penalty_selector_L1 = SelectFromModel(LogisticRegression(penalty="l1",
                                                             C=0.1,
                                                             solver='liblinear'))
    # 报错，指定求解器，错误就会消失。l1 支持 ‘liblinear’ 和 ‘saga’ L2 处理 newton-cg’、’lbfgs’、’sag’ 和 ‘saga’
    penalty_selector_array_L1 = penalty_selector_L1.fit_transform(df_noNA, y)
    # 特征选择embedded-基于惩罚项的特征选择法，将选择后的特征转换为DataFrame
    X_selected_embedded_penalty_L1 = pd.DataFrame(penalty_selector_array_L1,
                                                  columns=df_copy.columns[
                                                      penalty_selector_L1.get_support(indices=True)])

    # 特征选择embedded-基于惩罚项的特征选择法，L1选择的特征保存为文件
    X_selected_embedded_penalty_L1.to_excel(os.path.join(result_folder,subfolders['filter_based_on_L1_folder'],'选择的特征-embedded-penalty_L1.xlsx'))
    st.success("基于L1的embedded特征选择方法已完成!")

def feature_selection_embedded_based_on_L1L2():
    """
    基于L1和L2惩罚项的特征选择法
    :return:
    """
    # 特征选择embedded-基于惩罚项的特征选择法，L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要，可结合L2惩罚项来优化，操作如下：
    # 若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，将这一集合中的特征平分L1中的权值，故需要构建一个新的逻辑回归模型。
    # 特征选择embedded-基于惩罚项的特征选择法，带L1和L2惩罚项的逻辑回归作为基模型的特征选择，参数threshold为权值系数之差的阈值
    penalty_selector_L1L2 = SelectFromModel(LR(threshold=0.5, C=0.1))
    penalty_selector_array_L1L2 = penalty_selector_L1L2.fit_transform(df_noNA, y)
    # 特征选择embedded-基于惩罚项的特征选择法，将选择后的特征转换为DataFrame
    X_selected_embedded_penalty_L1L2 = pd.DataFrame(penalty_selector_array_L1L2, columns=df_copy.columns[
        penalty_selector_L1L2.get_support(indices=True)])

    # 特征选择embedded-基于惩罚项的特征选择法，L1L2选择的特征保存为文件
    X_selected_embedded_penalty_L1L2.to_excel(os.path.join(result_folder,subfolders['filter_based_on_L1L2_folder'],'选择的特征-embedded-penalty_L1和L2.xlsx'))
    st.success("基于L1和L2的embedded特征选择方法已完成!")

def feature_selection_embedded_based_on_SVM_L1():
    """
    使用 feature_selection 库的 SelectFromModel 类结合 SVM 模型
    :return:
    """
    lsvc_selector = LinearSVC(C=0.01, penalty='l1', dual=False).fit(df_noNA, y)
    lsvc_model = SelectFromModel(lsvc_selector, prefit=True)
    X_sfm_svm_array = lsvc_model.transform(df_noNA)
    # 将选择后的特征转换为DataFrame
    X_selected_svm = pd.DataFrame(X_sfm_svm_array, columns=df_copy.columns[lsvc_model.get_support(indices=True)])

    # 特征选择SVM和penalty_L1，选择的特征保存为文件
    X_selected_svm.to_excel(os.path.join(result_folder,subfolders['filter_based_on_SVM_L1_folder'],'选择的特征-SVM和penalty_L1.xlsx'))
    st.success("基于L1和SVM的embedded特征选择方法已完成!")

def feature_selection_embedded_based_on_GBDT():
    """
    特征选择embedded-基于树模型的特征选择法,树模型中GBDT也可用来作为基模型进行特征选择
    :return:
    """
    # 特征选择embedded-基于树模型的特征选择法，GBDT作为基模型的特征选择
    gbdt_selector = SelectFromModel(GradientBoostingClassifier())
    gbdt_selector_array = gbdt_selector.fit_transform(df_noNA, y)
    # 特征选择embedded-基于树模型的特征选择法，将选择后的特征转换为DataFrame
    X_selected_embedded_gbdt = pd.DataFrame(gbdt_selector_array,
                                            columns=df_copy.columns[gbdt_selector.get_support(indices=True)])

    # 特征选择embedded-GBDT，选择的特征保存为文件
    X_selected_embedded_gbdt.to_excel(os.path.join(result_folder,subfolders['filter_base_on_GBDT_folder'],'选择的特征-embedded-GBDT.xlsx'))
    st.success("基于GBDT的embedded特征选择方法已完成!")

def feature_selection_embedded_based_on_Lasso():
    """
    特征选择embedded-基于Lasso的特征选择法
    :return:
    """
    # 区分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(df_copy, y, random_state=100)

    # 构造不同的lambda值
    Lambdas = np.logspace(-5, 2, 200)

    # 设置交叉验证的参数，使用均方误差评估
    lasso_cv = LassoCV(alphas=Lambdas, cv=10, max_iter=10000)
    lasso_cv.fit(X_train, Y_train)

    # 基于最佳lambda值建模
    lasso_selector = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)  # 删除normalize

    lasso_selector.fit(X_train, Y_train)

    # 模型评估
    lasso_pred = lasso_selector.predict(X_test)

    # 均方误差
    MSE = mean_squared_error(Y_test, lasso_pred)

    # 输出每个特征的系数
    output_lasso_coef = pd.DataFrame({'Feature': X_train.columns, 'coefficient': lasso_selector.coef_.tolist()})
    output_lasso_coef.to_excel(os.path.join(result_folder,subfolders['filter_based_on_Lasso_folder'],'全部特征的系数-lasso.xlsx'))

    # 固定alpha，训练Lasso模型，展示每个特征的预测R2
    alpha = 0.1
    lasso_fix_alpha = Lasso(alpha=alpha)
    y_pred_lasso_fix_alpha = lasso_fix_alpha.fit(X_train, Y_train).predict(X_test)
    r2_score_lasso_fix_alpha = r2_score(Y_test, y_pred_lasso_fix_alpha)

    plt.plot(lasso_fix_alpha.coef_, color='gold', linewidth=2, label='Lasso coefficients')
    plt.title(f"Lasso R^2: {r2_score_lasso_fix_alpha}")
    plt.savefig(os.path.join(result_folder,subfolders['filter_based_on_Lasso_folder'],'lasso_coefficients.png'))
    st.success("基于LASSO的embedded特征选择方法已完成!")

def feature_importance_estimate():
    """
    特征重要性评估
    特征选择，排列重要性评估，展示重要性变化
    :return:
    """
    # 区分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(df_noNA, y, random_state=100)

    # 用随机森林做模型
    Permutation_selector = RandomForestClassifier(n_estimators=100,
                                                  bootstrap=True,
                                                  max_features='sqrt')  # bootstrap

    # 在训练集拟合模型
    Permutation_selector_array = Permutation_selector.fit(X_train, Y_train)

    # 作图展示重要性
    # perm = PermutationImportance(Permutation_selector, random_state=10).fit(X_test, Y_test)
    # html_content = eli5.show_weights(perm, feature_names=df_copy.columns.tolist())
    # 生成的html文件，可在浏览器中打开。(成功)
    # 生成的html文件转成image的方式。（失败）
    # with open(os.path.join(result_folder,subfolders['feature_importance_folder'],'feature_importance.html'), 'w', encoding='gbk') as f:
    #     f.write(html_content.data)

    # 特征选择，排列重要性评估，展示重要性下降
    # 区分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(df_noNA, y, random_state=100)

    # 用随机森林做模型
    perm_RandomForest_selector = RandomForestClassifier(n_estimators=100, random_state=1)

    # 在训练集拟合模型
    perm_RandomForest_selector.fit(X_train, Y_train)

    # 在测试集计算得分作为baseline
    permutation_baseline = perm_RandomForest_selector.score(X_test, Y_test)

    # 在测试集改变特征的顺序10次，计算每次的得分、得分的均值和标准差
    permutation_result = permutation_importance(perm_RandomForest_selector,
                                                X_test,
                                                Y_test,
                                                n_repeats=10,
                                                random_state=1,
                                                scoring='accuracy')
    importances = permutation_result.importances_mean

    # 输出每个特征多次改变顺序的得分的平均值
    output_permutation_importances = pd.DataFrame(
        {'Feature': df_copy.columns, 'importances_mean': importances})

    # 特征选择，排列重要性评估，每个特征的得分的平均值保存为文件
    output_permutation_importances.to_excel(os.path.join(result_folder,subfolders['feature_importance_folder'],'选择的特征的衡量指标-排列重要性评估-多次排序的得分平均值.xlsx'))

    # 画图，运行结果，重要性下降的越多，说明该特征越重要
    plt.figure(figsize=(20, 8))
    sns.barplot(x='importances_mean', y='Feature', data=output_permutation_importances)
    plt.title('Feature importances_mean', fontsize=20)
    plt.xlabel('importances_mean', fontsize=20)
    plt.ylabel('Feature', fontsize=3)
    plt.savefig(os.path.join(result_folder,subfolders['feature_importance_folder'],'排列重要性评估-多次排序的得分平均值.png'))

    # 特征选择，重要性评估
    # 用极度随机树做模型，评估参数的重要性
    importances_selector = ExtraTreesClassifier()
    importances_selector.fit(df_noNA, y)

    X_importances = list(zip(df_copy.columns, importances_selector.feature_importances_))
    output_importances = pd.DataFrame(X_importances, columns=['feature', 'importances'])
    output_importances = output_importances.sort_values(by='importances', ascending=False).head(20)

    # 特征选择，每个特征的重要性保存为文件
    output_importances.to_excel(os.path.join(result_folder,subfolders['feature_importance_folder'],'选择的特征的衡量指标-重要性评估.xlsx'))


def data_read_and_process():
    """
    读取数据并处理
    :return:
    """
    file=os.path.join(data_folder,'egang_040305M04_格式化特征集-添加故障标记.xlsx')
    # 读入振动信号的FFT特征参数，或FFT特征参数的tsfresh生成的特征
    df = pd.read_excel(file)
    df_copy = df.copy()

    y = df['故障标记']  # 从原始数据获取故障列，因变量y

    columns = df_copy.columns  # 删除自变量X不需要的列
    if '设备编码' in columns:
        del df_copy['设备编码']
    if '故障标记' in columns:
        del df_copy['故障标记']
    if '信号时间' in columns:
        del df_copy['信号时间']
    if '诊断报告发出的时间' in columns:
        del df_copy['诊断报告发出的时间']
    if '故障时间分组' in columns:
        del df_copy['故障时间分组']
    return df, df_copy, y


def data_preprocessing(df_copy):
    """
    处理数据
    1、标准化、
    2、归一化、
    3、区间缩放、
    4、二值化
    5、缺失值处理

    :return:
    """
    # 数据预处理
    df_standard = StandardScaler().fit_transform(df_copy)

    # 区间缩放，返回值为[0,1]区间的数据
    df_MinMax = MinMaxScaler().fit_transform(df_copy)

    # 归一化，返回值为归一化后的数据
    df_normalize = Normalizer().fit_transform(df_copy)

    # 对数据进行二值化,二值化，设定一个阈值，例如3，大于阈值的赋值为1，反之为0，返回值为二值化后的数据
    df_binary = Binarizer(threshold=3).fit_transform(df_copy)

    # 参数missing_value为缺失值的表示形式，默认为NaN,参数strategy为缺失值填充方式，默认为mean
    null_numbers = df_copy.isnull().sum().sum()
    if null_numbers == 0:
        df_noNA = df_copy
    else:
        df_noNA = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df_copy)

    return df_noNA, df_standard, df_MinMax, df_normalize, df_binary


class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        # 权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight,
                                    random_state=random_state, solver=solver, max_iter=max_iter,
                                    multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        # 使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)
        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


if __name__ == '__main__':
    # 侧边栏-特征工程6大步骤
    create_sidebar()
    # 第一步：上传数据
    st.subheader('步骤1:输入数据路径')
    data_folder = st.text_input("请输入数据的文件夹路径：")

    if data_folder:
        # 检查路径是否存在
        if os.path.exists(data_folder):
            st.success(f"文件夹路径正确: {data_folder}")

            # 获取文件夹中所有后缀为'_格式化特征集.csv'的CSV文件
            csv_files = [f for f in os.listdir(data_folder) if f.endswith('_格式化特征集.csv')]

            if csv_files:
                # 读取全部后缀"_格式化特征集.csv"的文件
                # 如果循环读取，在st.checkbox('趋势图',key=f'趋势图')时会报错，因为参数key要求唯一
            # for uploaded_file in csv_files[:1]:
                uploaded_file=csv_files[0]
                # 判断文件类型并加载数据
                df = pd.read_csv(os.path.join(data_folder, uploaded_file))
                drop_col = df.filter(like='Unnamed: 0', axis='columns')
                df.drop(columns=drop_col, inplace=True)

                device_name = uploaded_file.split('.')[0][:-7]
                data_row = df.shape[0]
                data_col = df.shape[1]
                st.write(f"数据预览(数据行数:{data_row},数据列数:{data_col}):", df.head())  # 显示数据的前5行

                # 创建保存结果的文件夹
                # 定义基础路径
                result_folder = os.path.join(os.path.dirname(data_folder), 'result')

                # 定义各个子文件夹路径
                subfolders = {
                    'trend_dir': '可视化结果/趋势图',
                    'hist_dir': '可视化结果/直方图',
                    'scatter_dir': '可视化结果/散点图',
                    'box_dir': '可视化结果/箱型图',
                    'cluster_dir': '聚类结果',
                    'feature_generate_based_on_tsfresh':'特征生成结果/tsfresh',
                    'feature_generate_based_on_ECSM':'特征生成结果/ECSM',
                    'filter_variance_selection_folder': '特征选择结果/过滤法-方差选择法',
                    'filter_chi2_estimator_folder': '特征选择结果/过滤法-卡方检验',
                    'filter_correlation_coefficient_folder': '特征选择结果/过滤法-相关系数',
                    'filter_RFE_folder': '特征选择结果/包装法-RFE递归消除法',
                    'filter_based_on_L1_folder': '特征选择结果/嵌入法-基于L1惩罚项的特征选择法',
                    'filter_based_on_L1L2_folder': '特征选择结果/嵌入法-结合L1和L2惩罚项的特征选择法',
                    'filter_base_on_GBDT_folder': '特征选择结果/嵌入法-基于树模型的特征选择法',
                    'filter_based_on_SVM_L1_folder': '特征选择结果/嵌入法-结合SVM和L1惩罚项的特征选择',
                    'filter_based_on_Lasso_folder': '特征选择结果/嵌入法-LASSO',
                    'feature_importance_folder':'特征选择结果/特征重要性'
                }

                # 创建主结果文件夹
                os.makedirs(result_folder, exist_ok=True)

                # 循环创建子文件夹
                for folder_name, folder_path in subfolders.items():
                    full_path = os.path.join(result_folder, folder_path)
                    os.makedirs(full_path, exist_ok=True)

                # 第二步：数据分析和可视化
                st.subheader('步骤2:数据可视化')
                st.write('注意：存在几种特征则会生成几种对应的图形，但每种图形只有一个示例展示。')
                # toggle_state = st.toggle('注意!')
                # if toggle_state:
                #     st.write('每种图形只显示一个示例')
                # 数值列
                date_col = df.filter(like='时间').columns[0]
                device_col = df.filter(like='设备').columns[0]
                df_plot = df.loc[:, ~df.columns.str.contains('时间|设备', case=False)]
                df_plot.columns = df_plot.columns.str.replace('/', '_')
                col1, col2, col3, col4 = st.columns(4)
                # 生成趋势图、散点图、直方图、箱型图
                with col1:
                    if st.checkbox('趋势图', key=f'{device_name}_趋势图'):
                        # st.subheader('趋势图')
                        if len(df_plot.columns) >= 1:
                            for col in df_plot.columns[:1]:
                                fig_trend, ax_trend = plt.subplots()
                                ax_trend.plot(pd.to_datetime(df[date_col]), df_plot[col])
                                ax_trend.set_title(f'({col})的趋势图')
                                ax_trend.set_xlabel('日期')
                                ax_trend.set_ylabel(f'{col}')
                                trend_plot_path = os.path.join(result_folder,subfolders['trend_dir'], f'{col}趋势图.png')
                                fig_trend.savefig(trend_plot_path)
                                st.pyplot(fig_trend)
                                st.success(f"趋势图已保存至: {trend_plot_path}")
                with col2:
                    if st.checkbox('散点图', key=f'{device_name}_scatter'):
                        # st.subheader('散点图')
                        if len(df_plot.columns) >= 1:
                            for col in df_plot.columns[:1]:
                                fig_scatter, ax_scatter = plt.subplots()
                                ax_scatter.scatter(pd.to_datetime(df[date_col]), df_plot[col])
                                ax_scatter.set_title(f'{col}的散点图')
                                ax_scatter.set_xlabel('日期')
                                ax_scatter.set_ylabel(f'{col}')
                                scatter_plot_path = os.path.join(result_folder,subfolders['scatter_dir'], f'{col}散点图.png')
                                fig_scatter.savefig(scatter_plot_path)
                                st.pyplot(fig_scatter)
                                st.success(f"散点图已保存至: {scatter_plot_path}")
                with col3:
                    if st.checkbox('直方图', key=f'{device_name}_直方图'):
                        # 生成直方图
                        if len(df_plot.columns) >= 1:
                            for col in df_plot.columns[:1]:
                                fig_hist, ax_hist = plt.subplots(figsize=(20, 20))
                                ax_hist.hist(df_plot[col], bins=20)
                                ax_hist.set_title(f'{col}的直方图')
                                ax_hist.set_xlabel(f'{col}')
                                ax_hist.set_ylabel('频率')
                                ax_hist.grid(True)
                                hist_plot_path = os.path.join(result_folder,subfolders['hist_dir'], f'{col}直方图.png')
                                fig_hist.savefig(hist_plot_path)
                                st.pyplot(fig_hist)
                                st.success(f"直方图已保存至: {hist_plot_path}")
                with col4:
                    if st.checkbox('箱型图', key=f'{device_name}_箱型图'):
                        # 生成箱型图
                        if len(df_plot.columns) >= 1:
                            for col in df_plot.columns[:1]:
                                fig_box, ax_box = plt.subplots(figsize=(20, 20))
                                ax_box.boxplot(df_plot[col])
                                ax_box.set_title(f'{col}的箱型图')
                                ax_box.set_xlabel(f'变量{col}')
                                ax_box.set_ylabel('值')
                                ax_box.grid(True)
                                box_plot_path = os.path.join(result_folder,subfolders['box_dir'], f'{col}直方图.png')
                                fig_box.savefig(box_plot_path)
                                st.pyplot(fig_box)
                                st.success(f"箱型图已保存至: {box_plot_path}")

                # 第三步：数据过滤
                st.subheader("步骤3:数据清洗")
                if st.checkbox('数据清洗', key=f'{device_name}_数据清洗'):
                    filter_method = st.radio(
                        "过滤方法选择",
                        ("箱型图过滤", "3-Sigma过滤", "移动平均过滤"),
                    )

                    # 应用过滤方法
                    filtered_df = filter_data(df, filter_method)

                    # 不需要过滤和聚类的列
                    date_col = df.filter(like='时间').columns[0]
                    device_col = df.filter(like='设备').columns[0]

                    remaining_columns = [col for col in df.columns if col not in [date_col, device_col]]
                    exclusive_columns = [col for col in df.columns if '时间' in col or '编码' in col]

                    # 保存过滤后的数据
                    if st.button(f"保存({filter_method})", key=f'{filter_method}'):
                        filtered_folder = filter_method.replace(" ", "_").lower()
                        filtered_path=os.path.join(result_folder,'过滤结果',filtered_folder)
                        os.makedirs(filtered_path, exist_ok=True)
                        file_path = os.path.join(filtered_path,
                                                 f"{uploaded_file[:-11]}({filtered_folder}结果).xlsx")
                        if not filtered_df.empty:
                            filtered_df.to_excel(file_path, index=False)
                            st.success(
                                f"过滤后的数据(行数:{filtered_df.shape[0]},数据列数:{filtered_df.shape[1]}),结果已保存到目录:'{file_path}'")
                        else:
                            st.warning(f"结果未保存到:{file_path}")

                # 第四步：进行聚类操作
                # 不需要过滤和聚类的列
                date_col = df.filter(like='时间').columns[0]
                device_col = df.filter(like='设备').columns[0]

                remaining_columns = [col for col in df.columns if col not in [date_col, device_col]]
                exclusive_columns = [col for col in df.columns if '时间' in col or '编码' in col]
                st.subheader("步骤4:工况分割")
                if st.checkbox('进行聚类分析', key=f'{device_name}_聚类操作'):
                    # 聚类所需的数值型数据
                    if len(remaining_columns) > 0:
                        # 选择聚类数目
                        num_clusters = st.slider("选择聚类数量", 2, 10, 3)

                        # KMeans聚类
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                        df['Cluster'] = kmeans.fit_predict(df[remaining_columns])

                        # 显示聚类结果
                        st.subheader(f'聚类结果 (K={num_clusters})')
                        # st.write("数据预览",df.head())

                        # 将聚类结果保存到文件
                        clustered_file_path = os.path.join(result_folder,subfolders['cluster_dir'], 'clustered_data.csv')
                        df.to_csv(clustered_file_path, index=False)
                        st.success(f"聚类结果已保存至: {clustered_file_path}")

                        # 可视化聚类结果
                        st.subheader('聚类散点图')
                        fig_cluster, ax_cluster = plt.subplots()
                        sns.scatterplot(data=df, x=remaining_columns[0], y=remaining_columns[1], hue='Cluster',
                                        palette='tab10',
                                        ax=ax_cluster)
                        ax_cluster.set_title(f'聚类结果(K={num_clusters})')
                        cluster_plot_path = os.path.join(result_folder,subfolders['cluster_dir'], 'cluster_plot.png')
                        fig_cluster.savefig(cluster_plot_path)
                        st.pyplot(fig_cluster)
                        st.success(f"聚类散点图已保存至: {cluster_plot_path}")
                    else:
                        st.warning("没有足够的数值型数据进行聚类分析。")

                # 步骤5：特征生成
                """
                st.subheader("步骤5:特征生成")

                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox('tsfresh方法', key=f'{device_name}_tsfresh'):
                        # st.write('tsfresh方法简介')
                        record_result_all = []
                        for device in os.listdir(data_folder):
                            if device.endswith('_格式化特征集.csv'):
                                df = pd.read_csv(os.path.join(data_folder, device),
                                                 index_col=False)  # csv文件没有表头的时候，不用header参数

                                # 读取故障时间段
                                data = pd.read_excel(
                                    os.path.join(data_folder, '人工标记故障时段_20241010_训练集和测试集.xlsx'))

                                # 对时间列格式化为datetime
                                time_columns = [col for col in data.columns if '时间' in col]
                                for col in time_columns:
                                    data[col] = pd.to_datetime(data[col])

                                matching_rows = data[
                                    data['设备名称'] == device.replace('_格式化特征集.csv', '')]
                                equip_info = matching_rows.reset_index(drop=True)  # 复制故障时间的记录，用于输出全部结果
                                if len(matching_rows) > 0:
                                    for times in range(len(matching_rows)):
                                        # trouble_type = data.loc[times, '故障类型'] #原有的代码，全部设备的故障记录用times查询，故障类型匹配错误
                                        trouble_type = matching_rows['故障类型'].iloc[
                                            times]  # 改用本次循环匹配到的数据，获取故障类型的信息
                                        stamp_start = matching_rows['训练集开始时间'].iloc[times]
                                        stamp_end = matching_rows['训练集结束时间'].iloc[times]
                                        trouble_time = matching_rows['诊断报告发出的时间'].iloc[times]
                                        df['诊断报告发出的时间'] = trouble_time

                                        trouble_time_list = []
                                        datetime_start = ''
                                        datetime_end = ''
                                        for times1 in range(len(matching_rows)):
                                            if (matching_rows['故障类型'].iloc[times1] == trouble_type):
                                                datetime_start = matching_rows['故障开始时间'].iloc[times1]
                                                datetime_end = matching_rows['故障结束时间'].iloc[times1]
                                                trouble_time_list.append(
                                                    str(datetime_start) + ',' + str(datetime_end))

                                        # excel文档记录
                                        extracted_features = pd.DataFrame()
                                        features_filtered = pd.DataFrame()
                                        record_result = pd.DataFrame()
                                        data_nodup = pd.DataFrame()
                                        data_nodup_merge = pd.DataFrame()

                                        # 截取故障时段的数据
                                        df['信号时间'] = pd.to_datetime(df['信号时间'])
                                        new_df = df.loc[
                                                 (df.loc[:, '信号时间'] >= stamp_start) & (
                                                         df.loc[:, '信号时间'] < stamp_end),
                                                 :]

                                        length_new_df1 = len(new_df)

                                        # 去除列名的特殊字符
                                        new_df.rename(columns=lambda col: col.replace('"', ''), inplace=True)
                                        new_df.rename(columns=lambda col: col.replace('/', ''), inplace=True)

                                        length2 = 0
                                        for times2 in trouble_time_list:
                                            datetime_start1 = times2.split(',')[0]
                                            datetime_end1 = times2.split(',')[1]
                                            length2 += len(
                                                new_df.loc[(new_df.loc[:, '信号时间'] >= datetime_start1) & (
                                                        new_df.loc[:, '信号时间'] < datetime_end1), '信号时间'])

                                        # 输出每次故障的训练数据的信息
                                        record_result['设备名称'] = [equip_info['设备名称'][times]]
                                        record_result['报告发出时间'] = [trouble_time]
                                        record_result['训练集开始时间'] = [stamp_start]
                                        record_result['训练集结束时间'] = [stamp_end]
                                        record_result['故障开始时间'] = [
                                            matching_rows['故障开始时间'].iloc[times]]
                                        record_result['故障结束时间'] = [
                                            matching_rows['故障结束时间'].iloc[times]]
                                        record_result['训练集全部点数'] = [length_new_df1]
                                        record_result['故障时间段点数'] = [length2]

                                        if not os.path.exists(os.path.join(data_folder,
                                                                           device.replace('_格式化特征集.csv',
                                                                                          '') + '(报告发出时间：' + str(
                                                                               trouble_time)[
                                                                                                                   :10] + ')' + "tsfresh特征生成.xlsx")):
                                            if length2 > 30:
                                                # 将故障时间段设置为1
                                                new_df['故障'] = False
                                                datetime_start1 = ''
                                                datetime_end1 = ''
                                                for times2 in trouble_time_list:
                                                    datetime_start1 = times2.split(',')[0]
                                                    datetime_end1 = times2.split(',')[1]
                                                    new_df.loc[
                                                        (new_df.loc[:, '信号时间'] >= datetime_start1) & (
                                                                new_df.loc[:,
                                                                '信号时间'] < datetime_end1), '故障'] = True

                                                new_df_f = new_df  # 复制一个表格，用于特征生成和特征选择

                                                # 选择需要合并的列
                                                new_df_f['year'] = new_df_f['信号时间'].dt.year
                                                new_df_f['month'] = new_df_f['信号时间'].dt.month  # 按月统计
                                                new_df_f['week'] = new_df_f['信号时间'].apply(
                                                    lambda x: x.isocalendar()[1])  # 按周统计
                                                new_df_f['day'] = new_df_f['信号时间'].dt.day  # 按日统计
                                                cols_to_merge = ['year', 'week',
                                                                 '故障']  # 按月或按周或按天或按小时统计，选择一种统计方式，选择的字段写入该行，统计范围影响生成的特征和特征选择的结果
                                                # 定义连接字符
                                                join_char = '-'
                                                # 合并列并创建新列
                                                new_df_f['故障时间分组'] = new_df_f[cols_to_merge].astype(
                                                    str).agg(
                                                    lambda x: join_char.join(x),
                                                    axis=1)

                                                # 获取信号时间和故障标记，用于与特征选择结果的合并
                                                # 每个故障时间分组保留一个信号时间、故障时间，与生成的特征合并，做趋势展示按时间排序，且显示故障时间
                                                data_nodup = new_df_f.groupby('故障时间分组').head(1)
                                                data_nodup_merge = data_nodup[
                                                    ['信号时间', '故障时间分组', '设备编码',
                                                     '诊断报告发出的时间']].reset_index()
                                                data_nodup_merge.drop(columns='index', inplace=True)

                                                # TsFresh时间序列的特征工程，官网代码
                                                # 选取故障的信息作为特征选择的目标y
                                                df_y = new_df_f[
                                                    ['故障', '故障时间分组']].drop_duplicates().set_index(
                                                    '故障时间分组')  # 提取目标y的表格
                                                unique_count_y = len(
                                                    set(df_y['故障']))  # y的值必须是2个，true和false，如果只有一个值，不做特征工程

                                                if unique_count_y > 1:
                                                    columns = new_df_f.columns
                                                    if '设备编码' in columns:
                                                        del new_df_f['设备编码']  # 删除不需要生成特征的参数
                                                    if 'year' in columns:
                                                        del new_df_f['year']
                                                    if 'month' in columns:
                                                        del new_df_f['month']
                                                    if 'week' in columns:
                                                        del new_df_f['week']
                                                    if 'day' in columns:
                                                        del new_df_f['day']
                                                    if '故障' in columns:
                                                        del new_df_f['故障']
                                                    if '诊断报告发出的时间' in columns:
                                                        del new_df_f['诊断报告发出的时间']
                                                    if 'Unnamed: 0' in columns:
                                                        del new_df_f['Unnamed: 0']

                                                    extracted_features = extract_features(new_df_f,
                                                                                          column_id="故障时间分组",
                                                                                          column_sort="信号时间")  # 生成特征

                                                    df_y_sorted = df_y.reindex(
                                                        extracted_features.index.tolist())  # 目标y和生成的特征的index同步

                                                    # 特征选择，fdr_level应该设置为0.05，首先设置为0.1，如果没有选出符合条件的特征，放宽选择条件，0.1*1.5，
                                                    # 如果选出超过100个特征，限制选择条件，0.1*0.5
                                                    impute(extracted_features)  # 填补缺失值

                                                    fdr_level_1 = 0.1  # 首次特征选择
                                                    features_filtered_1 = select_features(extracted_features,
                                                                                          df_y_sorted['故障'],
                                                                                          fdr_level=fdr_level_1)

                                                    ##### 循环优化特征选择
                                                    col_count_1 = features_filtered_1.shape[1]

                                                    if col_count_1 == 0:
                                                        for i in range(1, 16, 1):
                                                            # print(i)
                                                            fdr_level_i = i / 20.0
                                                            features_filtered_i = select_features(
                                                                extracted_features,
                                                                df_y_sorted['故障'],
                                                                fdr_level=fdr_level_i)
                                                            col_count_i = features_filtered_i.shape[1]
                                                            if col_count_i > 0:
                                                                break
                                                        features_filtered_final = features_filtered_i
                                                        record_result['特征选择的批次'] = [i]
                                                        record_result['特征选择fdri'] = [fdr_level_i]
                                                        record_result['特征选择的个数i'] = [col_count_i]
                                                    elif col_count_1 > 0 and col_count_1 < 100:
                                                        features_filtered_final = features_filtered_1
                                                        record_result['特征选择的批次'] = [1]
                                                        record_result['特征选择fdri'] = [fdr_level_1]
                                                        record_result['特征选择的个数i'] = [col_count_1]
                                                    elif col_count_1 >= 100:
                                                        for i in range(1, 20, 1):
                                                            # print(i)
                                                            fdr_level_i = 0.05 / i
                                                            features_filtered_i = select_features(
                                                                extracted_features,
                                                                df_y_sorted['故障'],
                                                                fdr_level=fdr_level_i)
                                                            col_count_i = features_filtered_i.shape[1]
                                                            if col_count_i == 0:
                                                                if i == 1:
                                                                    fdr_level_i = fdr_level_1
                                                                    features_filtered_i = features_filtered_1
                                                                    col_count_i = col_count_1
                                                                else:
                                                                    fdr_level_i = 0.05 / (i - 1)
                                                                    features_filtered_i = select_features(
                                                                        extracted_features,
                                                                        df_y_sorted['故障'],
                                                                        fdr_level=fdr_level_i)
                                                                    col_count_i = features_filtered_i.shape[1]
                                                                break
                                                            elif col_count_1 > 0 and col_count_i < 100:
                                                                break
                                                        features_filtered_final = features_filtered_i
                                                        record_result['特征选择的批次'] = [i]
                                                        record_result['特征选择fdri'] = [fdr_level_i]
                                                        record_result['特征选择的个数i'] = [col_count_i]

                                                        # 输出特征生成和特征选择的结果
                                                    extracted_features['故障标记'] = df_y_sorted[
                                                        '故障']  # 添加故障标记和故障时间段的分组
                                                    extracted_features['故障时间分组'] = df_y_sorted.index
                                                    features_filtered_final['故障标记'] = df_y_sorted['故障']
                                                    features_filtered_final['故障时间分组'] = df_y_sorted.index
                                                    # 合并选择的特征和信号时间
                                                    extracted_features_merge = pd.merge(extracted_features,
                                                                                        data_nodup_merge,
                                                                                        how="left",
                                                                                        left_on=[
                                                                                            "故障时间分组"],
                                                                                        right_on=[
                                                                                            "故障时间分组"])
                                                    extracted_features_merge_nodup = extracted_features_merge.drop_duplicates(
                                                        subset='故障时间分组').reset_index()  # 去除合并后的重复行
                                                    extracted_features_merge_nodup.drop(columns=['index'],
                                                                                        inplace=True)

                                                    features_filtered_merge = pd.merge(features_filtered_final,
                                                                                       data_nodup_merge,
                                                                                       how="left",
                                                                                       left_on=["故障时间分组"],
                                                                                       right_on=[
                                                                                           "故障时间分组"])
                                                    features_filtered_merge_nodup = features_filtered_merge.drop_duplicates(
                                                        subset='故障时间分组').reset_index()  # 去除合并后的重复行
                                                    features_filtered_merge_nodup.drop(columns=['index'],
                                                                                       inplace=True)

                                                    # tsfresh特征生成
                                                    extracted_features_merge_nodup.to_excel(
                                                        os.path.join(data_folder,
                                                                     device.replace(
                                                                         '_格式化特征集.csv',
                                                                         '') + '(报告发出时间：' + str(
                                                                         trouble_time)[
                                                                                                  :10] + ')' + "tsfresh特征生成.xlsx"),
                                                        index=False)

                                                    # tsfresh特征选择循环细化
                                                    features_filtered_merge_nodup.to_excel(
                                                        os.path.join(data_folder,
                                                                     device.replace(
                                                                         '_格式化特征集.csv',
                                                                         '') + '(报告发出时间：' + str(
                                                                         trouble_time)[
                                                                                                  :10] + ')' + "tsfresh特征选择循环细化.xlsx"),
                                                        index=False)

                                        record_result_all.append(record_result)
                        # 每次故障的训练数据的信息.csv
                        pd.concat(record_result_all).to_csv(
                            os.path.join(data_folder, "每次故障的训练数据的信息.csv"),
                            encoding='utf-8-sig')

                        # 一批设备的特征选择的结果，批量作图，概率密度图，数据区分故障和正常
                        # data = pd.read_excel(os.path.join(data_folder,'文件名称及对应的故障时间段.xlsx'))
                        record_result_all = []

                        for device in os.listdir(data_folder):
                            if device.endswith('tsfresh特征选择循环细化.xlsx'):
                                df = pd.read_excel(os.path.join(data_folder, device),
                                                   index_col=False)  # csv文件没有表头的时候，不用header参数

                                col_count = df.shape[1]  # 特征选择的结果的列数=2，表示选择结果为空，不画图，
                                # excel文档记录
                                record_result = pd.DataFrame()
                                record_result['设备名称'] = [device]
                                record_result['特征选择的个数'] = [col_count]
                                record_result_all.append(record_result)

                                if col_count > 5:
                                    # 概率密度图
                                    if not os.path.exists(
                                            os.path.join(data_folder,
                                                         device.replace('tsfresh特征选择循环细化.xlsx',
                                                                        '') + '概率密度图')):
                                        os.makedirs(
                                            os.path.join(data_folder,
                                                         device.replace('tsfresh特征选择循环细化.xlsx',
                                                                        '') + '概率密度图'))
                                        fig = plt.figure(figsize=(30, 20))

                                        # 去除列名的特殊字符
                                        df.rename(columns=lambda col: col.replace('"', ''), inplace=True)
                                        df.rename(columns=lambda col: col.replace('/', ''), inplace=True)
                                        df.drop(columns=['故障时间分组', '信号时间', '设备编码',
                                                         '诊断报告发出的时间'],
                                                inplace=True)
                                        columns = df.columns  # 提取列名
                                        value = '故障标记'
                                        columns = [item for item in columns if item != value]

                                        for column in columns:
                                            df_1col = df[['故障标记', column]]

                                            df_data1 = df_1col[df_1col['故障标记'] == True]
                                            df_data2 = df_1col[df_1col['故障标记'] == False]

                                            data1 = df_data1[column]
                                            data2 = df_data2[column]

                                            plt.figure(figsize=(20, 12))

                                            sns.kdeplot(data1, fill=True, color='blue', alpha=0.5, label='故障',
                                                        linewidth=2)
                                            sns.kdeplot(data2, fill=True, color='orange', alpha=0.5,
                                                        label='正常',
                                                        linewidth=2)

                                            plt.legend(fontsize=20)
                                            plt.tick_params(axis='both', labelsize=20)
                                            plt.title(f'{column}', fontsize=20)
                                            plt.xlabel('Value', fontsize=20)
                                            plt.ylabel('Density', fontsize=20)
                                            plt.xticks(rotation=30)
                                            image_file = os.path.join(data_folder,
                                                                      device.replace(
                                                                          'tsfresh特征选择循环细化.xlsx',
                                                                          '') + '概率密度图',
                                                                      column + '.png')
                                            plt.savefig(image_file, dpi=300)
                                            plt.close()  # 关闭当前图形，为下一个图形腾出空间

                        pd.concat(record_result_all).to_csv(
                            os.path.join(data_folder, "特征选择作图的信息_循环优化_概率密度图.csv"),
                            encoding='utf-8-sig')

                        # 一批设备的特征选择的结果，批量作图，散点图，趋势图

                        record_result_all = []

                        for device in os.listdir(data_folder):
                            if device.endswith('tsfresh特征选择循环细化.xlsx'):
                                df = pd.read_excel(os.path.join(data_folder, device),
                                                   index_col=False)  # csv文件没有表头的时候，不用header参数
                                df['信号时间'] = pd.to_datetime(df['信号时间'])  # 设置信号时间的格式并排序
                                df = df.sort_values('信号时间', ascending=True)
                                trouble_time = df['诊断报告发出的时间'][0]  # 获取诊断报告发出的时间

                                col_count = df.shape[1]  # 特征选择的结果的列数=2，表示选择结果为空，不画图，
                                # excel文档记录
                                record_result = pd.DataFrame()
                                record_result['设备编码'] = df['设备编码'][0]
                                record_result['特征选择的个数'] = [col_count]
                                record_result_all.append(record_result)

                                if col_count > 5:
                                    # 创建文件夹
                                    if not os.path.exists(
                                            os.path.join(data_folder,
                                                         device.replace('tsfresh特征选择循环细化.xlsx',
                                                                        '') + '趋势图')):
                                        os.makedirs(
                                            os.path.join(data_folder,
                                                         device.replace('tsfresh特征选择循环细化.xlsx',
                                                                        '') + '趋势图'))
                                        fig = plt.figure(figsize=(30, 20))

                                        # 去除列名的特殊字符
                                        df.rename(columns=lambda col: col.replace('"', ''), inplace=True)
                                        df.rename(columns=lambda col: col.replace('/', ''), inplace=True)
                                        df.drop(columns=['故障时间分组', '故障标记', '设备编码',
                                                         '诊断报告发出的时间'],
                                                inplace=True)

                                        columns = df.columns  # 提取列名
                                        value = '信号时间'
                                        columns = [item for item in columns if item != value]

                                        for column in columns:
                                            plt.figure(figsize=(20, 12))
                                            max1 = df[column].max()

                                            plt.scatter(pd.to_datetime(df['信号时间']), df[column],
                                                        cmap='viridis',
                                                        alpha=0.7)
                                            plt.plot(pd.to_datetime(df['信号时间']), df[column])
                                            plt.axvline(trouble_time, color='red', linestyle='--',
                                                        label='报告发出时间')
                                            plt.text(trouble_time, max1, str(trouble_time)[:10], rotation=0,
                                                     verticalalignment='center',
                                                     horizontalalignment='center', fontsize=30)

                                            #                    plt.legend(fontsize =20)
                                            plt.tick_params(axis='both', labelsize=20)
                                            plt.title(f'{column}', fontsize=20)
                                            plt.xlabel('time', fontsize=20)
                                            plt.ylabel('value', fontsize=20)
                                            plt.xticks(rotation=30)
                                            image_file = os.path.join(data_folder,
                                                                      device.replace(
                                                                          'tsfresh特征选择循环细化.xlsx',
                                                                          '') + '趋势图',
                                                                      column + '.png')
                                            plt.savefig(image_file, dpi=300)
                                            #                    plt.savefig(f'{column}.png')
                                            plt.close()  # 关闭当前图形，为下一个图形腾出空间

                        pd.concat(record_result_all).to_excel(
                            os.path.join(data_folder, "特征选择作图的信息_循环优化_趋势图.xlsx"),
                            index=False)

                        # 无监督学习，凝聚层次聚类
                        record_result_all = []

                        for device in os.listdir(data_folder):
                            if device.endswith('_格式化特征集.csv'):
                                real_data = pd.read_csv(os.path.join(data_folder, device),
                                                        index_col=False)  # csv文件没有表头的时候，不用header参数

                                # 去除列名的特殊字符
                                real_data.rename(columns=lambda col: col.replace('"', ''), inplace=True)
                                real_data.rename(columns=lambda col: col.replace('/', ''), inplace=True)

                                # 删除不需要聚类的列
                                real_data.drop(columns=['设备编码', 'Unnamed: 0'], inplace=True)

                                columns = real_data.columns  # 提取列名
                                value = '信号时间'  # 数据保留故时间，用于聚类结果按时间作图，但列名删除故障标记
                                columns = [item for item in columns if item != value]

                                # 聚类
                                if not os.path.exists(
                                        os.path.join(data_folder,
                                                     device.replace('_格式化特征集.csv', '') + '聚类图')):
                                    os.makedirs(
                                        os.path.join(data_folder,
                                                     device.replace('_格式化特征集.csv', '') + '聚类图'))
                                    fig = plt.figure(figsize=(30, 20))
                                    for column in columns:
                                        df_2col = real_data[['信号时间', column]]  # 选择一列
                                        df_1col = real_data[[column]]  # 选择一列
                                        real_data_x = df_1col.values.reshape(-1, 1)  # 转为array

                                        # 计算层次聚类的链接矩阵
                                        Z = linkage(real_data_x, method='ward')

                                        # 根据距离阈值提取簇
                                        max_d = 50  # 距离阈值
                                        clusters = fcluster(Z, max_d, criterion='distance')

                                        # 可视化聚类结果
                                        plt.figure()
                                        plt.title(f'{column}')
                                        plt.xlabel('time')
                                        plt.ylabel('variable')
                                        plt.scatter(pd.to_datetime(df_2col['信号时间']), real_data_x[:, 0],
                                                    c=clusters,
                                                    cmap='viridis',
                                                    marker='o',
                                                    edgecolor='k', s=100)
                                        image_file = os.path.join(data_folder,
                                                                  device.replace('_格式化特征集.csv',
                                                                                 '') + '聚类图',
                                                                  column + '.png')
                                        plt.xticks(rotation=30)
                                        plt.savefig(image_file, dpi=300)
                                        plt.close()  # 关闭当前图形，为下一个图形腾出空间
                        st.success("tsfresh特征生成和选择已完成!")

                with col2:
                    if st.checkbox('ECSM(Exceedance Combination Selection Model)方法',
                                   key=f'{device_name}_ECSM'):
                        # st.write('ECS方法简介')
                        for device in os.listdir(data_folder):
                            if device.endswith('_格式化特征集.csv'):
                                try:
                                    df = pd.read_csv(os.path.join(data_folder, device),
                                                     index_col=False)  # csv文件没有表头的时候，不用header参数
                                except (ParserError, ValueError) as e:
                                    df = pd.DataFrame()
                                    # print("遇到解析错误，但程序继续执行,错误文件名为：" + device)

                                # 生成特征组合
                                columns = df.columns[4:].tolist()
                                # 如果‘诊断报告发出的时间’在列表中，则删除
                                if '诊断报告发出的时间' in columns:
                                    columns.remove('诊断报告发出的时间')

                                combinations_list = list(combinations(columns, 2))
                                length_df = len(df)
                                df['信号时间'] = pd.to_datetime(df['信号时间'])

                                total_rows = []
                                for j in columns:
                                    min1 = df.loc[:, j].median() / 5
                                    less_than_median_indices = [i for i in df.index if df.loc[i, j] < min1]
                                    total_rows = total_rows + less_than_median_indices
                                unique_list = list(set(total_rows))
                                # print("预计删除的点数：", len(unique_list))

                                # 设置故障时间段
                                data = pd.read_excel(
                                    os.path.join(data_folder, '人工标记故障时段_20241010_训练集和测试集.xlsx'))
                                time_columns = [col for col in data.columns if '时间' in col]
                                for col in time_columns:
                                    data[col] = pd.to_datetime(data[col])

                                matching_rows = data[
                                    data['设备名称'] == device.replace('_格式化特征集.csv', '')]
                                # 将在故障时间段的数据序号添加到列表中
                                fault_time_list = []
                                for times in range(len(matching_rows)):
                                    # 将故障时间段的序列号提取出来
                                    for i in range(len(df)):
                                        if df.loc[i, '信号时间'] >= matching_rows['故障开始时间'].iloc[
                                            times] and \
                                                df.loc[
                                                    i, '信号时间'] <= \
                                                matching_rows['故障结束时间'].iloc[times]:
                                            fault_time_list.append(i)
                                    # print("故障时间点数：", len(fault_time_list))

                                    # 将故障时间段从unique_list中删除
                                    if len(fault_time_list) < 200 or (
                                            len(fault_time_list) < (len(unique_list) / 4) and len(
                                        fault_time_list) > 200):
                                        for i in fault_time_list:
                                            if i in unique_list:
                                                unique_list.remove(i)
                                        # print("现在删除的点数：", len(unique_list))

                                # print("实际删除的点数：", len(unique_list))

                                df = pd.DataFrame(df.drop(unique_list))
                                # print(len(df), device)

                                for combination in combinations_list:
                                    column1 = combination[0]
                                    column2 = combination[1]
                                    if column1[3:] == column2[3:]:
                                        # df[column1 + "+" + column2] = df[column1] + df[column2]
                                        df[column1 + "-" + column2] = df[column1] - df[column2]
                                        df[column1 + " 除 " + column2] = df[column1] / df[column2]
                                        df[column2 + "-" + column1] = df[column2] - df[column1]
                                        df[column2 + " 除 " + column1] = df[column2] / df[column1]
                                    elif column1[3:] == column2[3:]:
                                        # df[column1 + "+" + column2] = df[column1] + df[column2]
                                        df[column1 + "-" + column2] = df[column2] - df[column1]
                                        df[column1 + " 除 " + column2] = df[column1] / df[column2]
                                        df[column2 + "-" + column1] = df[column2] - df[column1]
                                        df[column2 + " 除 " + column1] = df[column2] / df[column1]
                                length = len(df.columns)

                                # 设置阈值
                                length1 = 5
                                columns = df.columns[4:].tolist()
                                # 如果‘诊断报告发出的时间’在列表中，则删除
                                if '诊断报告发出的时间' in columns:
                                    columns.remove('诊断报告发出的时间')
                                df.index = range(len(df))
                                for column1 in columns:
                                    median1 = round(df[column1].quantile(0.5), 2)
                                    max1 = round(df[column1].max(), 2)
                                    step1 = (max1 - median1) / length1

                                    for a_threshold in range(1, length1):
                                        a_threshold1 = round(median1 + a_threshold * step1, 2)
                                        column1_1 = str(column1 + '≥' + str(a_threshold1))
                                        df[column1_1] = df[column1] >= a_threshold1
                                        # print(column1_1,a_threshold1)
                                # print(len(df.columns), df.columns)
                                # 设置故障时间段
                                if len(matching_rows) > 0:
                                    for times in range(len(matching_rows)):
                                        # datetime_start = matching_rows['开始时间'].iloc[times]
                                        # datetime_end = matching_rows['结束时间'].iloc[times]
                                        stamp_start = matching_rows['训练集开始时间'].iloc[times]
                                        stamp_end = matching_rows['训练集结束时间'].iloc[times]
                                        test_start = matching_rows['测试集开始时间'].iloc[times]
                                        test_end = matching_rows['测试集结束时间'].iloc[times]
                                        trouble_time = matching_rows['诊断报告发出的时间'].iloc[times]
                                        trouble_name = matching_rows['故障类型'].iloc[times]
                                        # print(trouble_name)
                                        # print(matching_rows)

                                        trouble_time_list = []
                                        datetime_start = ''
                                        datetime_end = ''
                                        for times1 in range(len(matching_rows)):
                                            if matching_rows['故障类型'].iloc[times1] == trouble_name:
                                                datetime_start = matching_rows['故障开始时间'].iloc[times1]
                                                datetime_end = matching_rows['故障结束时间'].iloc[times1]
                                                trouble_time_list.append(
                                                    str(datetime_start) + ',' + str(datetime_end))
                                                # print(matching_rows['故障类型'].iloc[times1])
                                                # print(trouble_time_list)

                                        # 将在stamp_start和stamp_end之间的数据进行训练，在test_start和test_end之间的数据进行测试
                                        new_df = pd.DataFrame(
                                            df.loc[
                                            (df.loc[:, '信号时间'] >= stamp_start) & (
                                                    df.loc[:, '信号时间'] < stamp_end),
                                            :])
                                        test_df = pd.DataFrame(
                                            df.loc[
                                            (df.loc[:, '信号时间'] >= test_start) & (
                                                    df.loc[:, '信号时间'] < test_end),
                                            :])

                                        if os.path.exists(os.path.join(data_folder,
                                                                       device.replace('_格式化特征集.csv',
                                                                                      '') + '(报告发出时间：' + str(
                                                                           trouble_time)[
                                                                                                               :10] + ')' + "特征筛选报告(训练集).xlsx")):
                                            data_old = pd.read_excel(
                                                os.path.join(data_folder, device.replace('_格式化特征集.csv',
                                                                                         '') + '(报告发出时间：' + str(
                                                    trouble_time)[:10] + ')' + "特征筛选报告(训练集).xlsx"))
                                            # 创建一个空的list对象
                                            combinations_list_finished = []
                                            for i in range(len(data_old)):
                                                column1 = str(data_old['特征1'].iloc[i]) + '≥' + str(
                                                    data_old['阈值1'].iloc[i])
                                                column2 = str(data_old['特征2'].iloc[i]) + '≥' + str(
                                                    data_old['阈值2'].iloc[i])
                                                column3 = str(data_old['特征3'].iloc[i]) + '≥' + str(
                                                    data_old['阈值3'].iloc[i])
                                                column4 = str(data_old['特征4'].iloc[i]) + '≥' + str(
                                                    data_old['阈值4'].iloc[i])
                                                tuple1 = (column1, column2, column3, column4)
                                                combinations_list_finished.append(tuple1)
                                            # print(combinations_list_finished)

                                        local1 = 0
                                        # excel文档记录
                                        data_save = pd.DataFrame(
                                            columns=['特征1', '阈值1', '特征2', '阈值2', '特征3', '阈值3',
                                                     '特征4',
                                                     '阈值4',
                                                     'TN(无故障，预测为无故障)', 'TP(故障，预测为故障)',
                                                     'FN(故障，预测为无故障)',
                                                     'FP(无故障，预测为故障)', '准确率((TP+TN)/(TP+FN+FP+TN))',
                                                     '精确率(TP/(TP+FP))',
                                                     '召回率(TP/(TP+FN))', '漏报率（FN/(FN+TP)）',
                                                     '真负率（TN/(FP+TN)）',
                                                     'F1 （(2 * Precision * Recall) / ( Precision + Recall))）',
                                                     'MCC (TP * TN - FP * FN) / ( ( ( TP + FP) * (TP + FN)*(TN + FP) * (TN + FN) ) **0.5 )'])

                                        # 生成特征组合
                                        columns = df.columns[length + 1:-1].tolist()
                                        # print(len(columns), columns)
                                        num_combination = 4
                                        # if '故障' in columns:
                                        #     columns.remove('故障')
                                        combinations_list = list(combinations(columns, num_combination))
                                        combination = combinations_list[0]
                                        # print(combination,type(combinations_list))

                                        # 将故障时间段设置为1
                                        new_df['故障'] = 0
                                        datetime_start1 = ''
                                        datetime_end1 = ''
                                        # print('样本时间：', stamp_start, stamp_end)
                                        # print(trouble_time_list)
                                        for times2 in trouble_time_list:
                                            datetime_start1 = times2.split(',')[0]
                                            datetime_end1 = times2.split(',')[1]
                                            new_df.loc[(new_df.loc[:, '信号时间'] >= datetime_start1) & (
                                                    new_df.loc[:, '信号时间'] < datetime_end1), '故障'] = 1
                                            # print("故障时间：", datetime_start1, datetime_end1)

                                        # 将combinations_list_finished的元素从combinations_list中删除
                                        # if 'combinations_list_finished' in locals():
                                        #     for combination in combinations_list_finished:
                                        #         print(combination)
                                        #         if combination in combinations_list:
                                        #             combinations_list.remove(combination)
                                        for combination in combinations_list:
                                            column1 = combination[0]
                                            column2 = combination[1]
                                            column3 = combination[2]
                                            column4 = combination[3]
                                            column1_1 = column1.split('≥')[0]
                                            column2_1 = column2.split('≥')[0]
                                            column3_1 = column3.split('≥')[0]
                                            column4_1 = column4.split('≥')[0]

                                            # 将变量放入一个列表
                                            variables = [column1_1, column2_1, column3_1, column4_1]
                                            # j += 1
                                            # 检查列表中是否存在重复元素
                                            if len(set(variables)) == len(
                                                    variables) and '故障' not in combination and '预测结果' not in combination:
                                                # print(combination)
                                                # print(type(combinations_list),type(combination))
                                                new_df['预测结果'] = (new_df[column1] & new_df[column2]) | (
                                                        new_df[column3] & new_df[column4])
                                                new_df['预测结果'] = new_df['预测结果'].map({True: 1, False: 0})
                                                cm = confusion_matrix(new_df['故障'], new_df['预测结果'])
                                                total = cm.sum()
                                                TN = cm[0][0]
                                                TP = cm[1][1]
                                                FN = cm[0][1]
                                                FP = cm[1][0]

                                                Accuracy = round((TP + TN) / total, 4)
                                                Precision = round(TP / (FP + TP), 4)
                                                Recall = round(TP / (FN + TP), 4)
                                                Miss_rate = round(FN / (FN + TP), 4)
                                                Specificity = round(TN / (TN + FP), 4)
                                                if (Precision + Recall) == 0:
                                                    F1 = 0
                                                else:
                                                    F1 = round((2 * Precision * Recall) / (Precision + Recall),
                                                               4)
                                                if (((TP + FP) * (TP + FN) * (TN + FP) * (
                                                        TN + FN)) ** 0.5) == 0:
                                                    MCC = 0
                                                else:
                                                    MCC = round((TP * TN - FP * FN) / (
                                                            ((TP + FP) * (TP + FN) * (TN + FP) * (
                                                                    TN + FN)) ** 0.5),
                                                                4)

                                                # print(cm)
                                                # print(Accuracy, Precision, Recall)

                                                # 将混淆矩阵添加到 Excel 文档中
                                                local1 += 1
                                                # print(column1, column2, column3, column4)
                                                new_row = {'特征1': column1.split('≥')[0],
                                                           '阈值1': column1.split('≥')[1],
                                                           '特征2': column2.split('≥')[0],
                                                           '阈值2': column2.split('≥')[1],
                                                           '特征3': column3.split('≥')[0],
                                                           '阈值3': column3.split('≥')[1],
                                                           '特征4': column4.split('≥')[0],
                                                           '阈值4': column4.split('≥')[1],
                                                           'TN(无故障，预测为无故障)': TN,
                                                           'TP(故障，预测为故障)': TP,
                                                           'FN(故障，预测为无故障)': FN,
                                                           'FP(无故障，预测为故障)': FP,
                                                           '准确率((TP+TN)/(TP+FN+FP+TN))': Accuracy,
                                                           '精确率(TP/(TP+FP))': Precision,
                                                           '召回率(TP/(TP+FN))': Recall,
                                                           '漏报率（FN/(FN+TP)）': Miss_rate,
                                                           '真负率（TN/(FP+TN)）': Specificity,
                                                           'F1 （(2 * Precision * Recall) / ( Precision + Recall))）': F1,
                                                           'MCC (TP * TN - FP * FN) / ( ( ( TP + FP) * (TP + FN)*(TN + FP) * (TN + FN) ) **0.5 )': MCC}
                                                data_save = data_save.append(new_row, ignore_index=True)

                                                gc.collect()
                                                if (local1 % 10000) == 1:
                                                    # print('已生成', local1, '个特征组合')
                                                    # 将data_save和data_old合并
                                                    if 'data_old' in locals():
                                                        data_save1 = pd.concat([data_save, data_old],
                                                                               ignore_index=True)
                                                    else:
                                                        data_save1 = data_save
                                                    data_save1.to_excel(
                                                        os.path.join(data_folder,
                                                                     device.replace('_格式化特征集.csv',
                                                                                    '') + '(报告发出时间：' + str(
                                                                         trouble_time)[
                                                                                                             :10] + ')' + "特征筛选报告(训练集).xlsx"),
                                                        index=False)
                                        # 保存 Word 文档
                                        data_save1.to_excel(os.path.join(data_folder,
                                                                         device.replace('_格式化特征集.csv',
                                                                                        '') + '(报告发出时间：' + str(
                                                                             trouble_time)[
                                                                                                                 :10] + ')' + "特征筛选报告(训练集).xlsx"),
                                                            index=False)
                                        data1 = data_save1.copy()
                                        # data1 = pd.read_excel(os.path.join(data_folder,
                                        #                                    device.replace('_格式化特征集.csv', '') + '(报告发出时间：' + str(
                                        #                                        trouble_time)[:10] + ')' + "特征筛选报告(训练集).xlsx"))

                                        data1_unique = data1.sort_values(
                                            by='F1 （(2 * Precision * Recall) / ( Precision + Recall))）',
                                            ascending=False).drop_duplicates(subset=['特征1'])
                                        data1_unique = data1_unique.drop_duplicates(subset=['特征2'])
                                        data1_unique = data1_unique.drop_duplicates(subset=['特征3'])
                                        data1_unique = data1_unique.drop_duplicates(subset=['特征4'])
                                        data1_unique.index = range(len(data1_unique))

                                        for i in range(300):
                                            column1 = data1_unique['特征1'][i] + "≥" + str(
                                                data1_unique['阈值1'][i])
                                            column2 = data1_unique['特征2'][i] + "≥" + str(
                                                data1_unique['阈值2'][i])
                                            column3 = data1_unique['特征3'][i] + "≥" + str(
                                                data1_unique['阈值3'][i])
                                            column4 = data1_unique['特征4'][i] + "≥" + str(
                                                data1_unique['阈值4'][i])

                                            for times2 in trouble_time_list:
                                                datetime_start1 = times2.split(',')[0]
                                                datetime_end1 = times2.split(',')[1]
                                                test_df.loc[(test_df.loc[:, '信号时间'] >= datetime_start1) & (
                                                        test_df.loc[:, '信号时间'] < datetime_end1), '故障'] = 1
                                                # print("故障时间：", datetime_start1, datetime_end1)

                                            test_df['预测结果'] = ((test_df[column1] >= data1_unique['阈值1'][
                                                i]) & (
                                                                           test_df[column2] >=
                                                                           data1_unique['阈值2'][i])) | (
                                                                          (test_df[column3] >=
                                                                           data1_unique['阈值3'][
                                                                               i]) & (
                                                                                  test_df[column4] >=
                                                                                  data1_unique['阈值4'][i]))
                                            test_df['预测结果'] = test_df['预测结果'].map({True: 1, False: 0})

                                            cm = confusion_matrix(test_df['故障'], test_df['预测结果'])
                                            total = cm.sum()
                                            TN = cm[0][0]
                                            TP = cm[1][1]
                                            FN = cm[0][1]
                                            FP = cm[1][0]

                                            Accuracy = round((TP + TN) / total, 4)
                                            Precision = round(TP / (FP + TP), 4)
                                            Recall = round(TP / (FN + TP), 4)
                                            Miss_rate = round(FN / (FN + TP), 4)
                                            Specificity = round(TN / (TN + FP), 4)
                                            if (Precision + Recall) == 0:
                                                F1 = 0
                                            else:
                                                F1 = round((2 * Precision * Recall) / (Precision + Recall), 4)
                                            if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) == 0:
                                                MCC = 0
                                            else:
                                                MCC = round((TP * TN - FP * FN) / (
                                                        ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5),
                                                            4)

                                            # print(cm)
                                            # print(Accuracy, Precision, Recall)

                                            if '测试集TN' not in data1_unique.columns:
                                                data1_unique['测试集TN'] = None
                                                data1_unique['测试集TP'] = None
                                                data1_unique['测试集FN'] = None
                                                data1_unique['测试集FP'] = None
                                                data1_unique['测试集准确率'] = None
                                                data1_unique['测试集精确率'] = None
                                                data1_unique['测试集召回率'] = None
                                                data1_unique['测试集漏报率'] = None
                                                data1_unique['测试集真负率'] = None
                                                data1_unique['测试集F1'] = None
                                                data1_unique['测试集MCC'] = None
                                            data1_unique['测试集TN'][i] = TN
                                            data1_unique['测试集TP'][i] = TP
                                            data1_unique['测试集FN'][i] = FN
                                            data1_unique['测试集FP'][i] = FP
                                            data1_unique['测试集准确率'][i] = Accuracy
                                            data1_unique['测试集精确率'][i] = Precision
                                            data1_unique['测试集召回率'][i] = Recall
                                            data1_unique['测试集漏报率'][i] = Miss_rate
                                            data1_unique['测试集真负率'][i] = Specificity
                                            data1_unique['测试集F1'][i] = F1
                                            data1_unique['测试集MCC'][i] = MCC

                                        data1_unique = data1_unique.iloc[:, :300]
                                        data1_unique.to_excel(os.path.join(data_folder,
                                                                           device.replace('_格式化特征集.csv',
                                                                                          '') + '(报告发出时间：' + str(
                                                                               trouble_time)[
                                                                                                                   :10] + ')' + "特征筛选报告(训练集+测试集).xlsx"),
                                                              index=False)
                        st.success("ECS特征生成和选择已完成!")
                """
                # 步骤6:特征选择
                st.subheader("步骤6:特征选择")
                df, df_copy, y=data_read_and_process()
                df_noNA, df_standard, df_MinMax, df_normalize, df_binary=data_preprocessing(df_copy)
                feature_importance_estimate()
                feature_selection_col1, feature_selection_col2, feature_selection_col3 = st.columns(3)
                # 过滤法
                with feature_selection_col1:
                    if st.checkbox('过滤法', key=f'{device_name}_过滤法'):
                        if st.checkbox('filter-方差选择法', key=f'{device_name}_filter-方差选择法'):
                            feature_selection_variance()
                        if st.checkbox('filter-卡方检验', key=f'{device_name}_filter-卡方检验'):
                            feature_selection_chi2_test()
                        if st.checkbox('filter-相关系数法', key=f'{device_name}_filter-相关系数法'):
                            feature_selection_correlation_coefficient()
                # 包装法
                with feature_selection_col2:
                    if st.checkbox('包装法', key=f'{device_name}_包装法'):
                        if st.checkbox('wrapper - RFE递归特征消除法',
                                       key=f'{device_name}_wrapper - RFE递归特征消除法'):
                            feature_selection_RFE()
                # 嵌入法
                with feature_selection_col3:
                    if st.checkbox('嵌入法', key=f'{device_name}_嵌入法'):
                        if st.checkbox('embedded - 基于L1惩罚项的特征选择法',
                                       key=f'{device_name}_embedded - 基于L1惩罚项的特征选择法'):
                            feature_selection_embedded_based_on_L1()
                        if st.checkbox('embedded - 结合L1和L2惩罚项的特征选择法',
                                       key=f'{device_name}_embedded - 结合L1和L2惩罚项的特征选择法'):
                            feature_selection_embedded_based_on_L1L2()
                        if st.checkbox('embedded - 基于树模型的特征选择法',
                                       key=f'{device_name}_embedded - 基于树模型的特征选择法'):
                            feature_selection_embedded_based_on_GBDT()
                        if st.checkbox('结合SVM和L1惩罚项的特征选择',
                                       key=f'{device_name}_结合SVM和L1惩罚项的特征选择'):
                            feature_selection_embedded_based_on_SVM_L1()
                        if st.checkbox('LASSO', key=f'{device_name}_结合LASSO的特征选择'):
                            feature_selection_embedded_based_on_Lasso()

            else:
                st.warning("文件夹中没有后缀为'_格式化特征集.csv'的CSV文件！")
        else:
            st.error("文件夹路径不存在，请重新输入！")
