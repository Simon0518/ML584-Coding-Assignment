import pandas
import statsmodels.api as stats
import sympy
import scipy
import numpy
import sklearn.ensemble as ensemble
import math


# In[2]:


def create_interaction(in_df1, in_df2):
    name1 = in_df1.columns
    name2 = in_df2.columns
    out_df = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            out_df[outName] = in_df1[col1] * in_df2[col2]
    return (out_df)


# In[3]:


def build_mnlogit(full_x, y, debug='N'):
    # Number of all parameters
    no_full_param = full_x.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    no_y_cat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(full_x.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    x = full_x.iloc[:, list(inds)]

    # The number of free parameters
    this_df = len(inds) * (no_y_cat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, x)
    this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    this_llk = logit.loglike(this_parameter.values)

    if (debug == 'Y'):
        print(this_fit.summary())
        print("Model Parameter Estimates:\n", this_parameter)
        print("Model Log-Likelihood Value =", this_llk)
        print("Number of Free Parameters =", this_df)

    # Recreat the estimates of the full parameters
    work_params = pandas.DataFrame(numpy.zeros(shape=(no_full_param, (no_y_cat - 1))))
    work_params = work_params.set_index(keys=full_x.columns)
    full_params = pandas.merge(work_params, this_parameter, how="left", left_index=True, right_index=True)
    full_params = full_params.drop(columns='0_x').fillna(0.0)

    # Return model statistics
    return (this_llk, this_df, full_params)


# In[22]:


purchase_likelihood = pandas.read_csv('Purchase_Likelihood.csv',
                       delimiter=',', usecols = ['insurance', 'group_size', 'homeowner', 'married_couple'])
purchase_likelihood = purchase_likelihood.dropna()

no_objs = purchase_likelihood.shape[0]

y = purchase_likelihood['insurance'].astype('category')

x_gs = pandas.get_dummies(purchase_likelihood[['group_size']].astype('category'))
x_ho = pandas.get_dummies(purchase_likelihood[['homeowner']].astype('category'))
x_mc = pandas.get_dummies(purchase_likelihood[['married_couple']].astype('category'))

# In[23]:


# Intercept only model
design_x = pandas.DataFrame(y.where(y.isnull(), 1))
llk0, df0, full_params0 = build_mnlogit(design_x, y, debug='Y')

# In[24]:


# Intercept + group_size
design_x = stats.add_constant(x_gs, prepend=True)
llk1_gs, df1_gs, full_params1_gs = build_mnlogit(design_x, y, debug='Y')
test_dev = 2 * (llk1_gs - llk0)
test_df = df1_gs - df0
test_p_value_gs = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value_gs)

# In[25]:


# Intercept + group_size + homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = stats.add_constant(design_x, prepend=True)
llk2_gs_ho, df2_gs_ho, full_params2_gs_ho = build_mnlogit(design_x, y, debug='Y')
test_dev = 2 * (llk2_gs_ho - llk1_gs)
test_df = df2_gs_ho - df1_gs
test_p_value_gs_ho = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value_gs_ho)

# In[26]:


# Intercept + group_size + homeowner + married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
design_x = stats.add_constant(design_x, prepend=True)
llk3_gs_ho_mc, df3_gs_ho_mc, full_params3_gs_ho_mc = build_mnlogit(design_x, y, debug='Y')
test_dev = 2 * (llk3_gs_ho_mc - llk2_gs_ho)
test_df = df3_gs_ho_mc - df2_gs_ho
test_p_value_gs_ho_mc = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value_gs_ho_mc)

# In[27]:


# Intercept + group_size + homeowner + married_couple + group_size * homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
llk4_gs_ho_mc_gsho, df4_gs_ho_mc_gsho, full_params4_gs_ho_mc_gsho = build_mnlogit(design_x, y, debug='Y')
test_dev = 2 * (llk4_gs_ho_mc_gsho - llk3_gs_ho_mc)
test_df = df4_gs_ho_mc_gsho - df3_gs_ho_mc
test_p_value_gs_ho_mc_gsho = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value_gs_ho_mc_gsho)

# In[28]:


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
design_x = design_x.join(x_homc)
design_x = stats.add_constant(design_x, prepend=True)
llk5_gs_ho_mc_gsho_homc, df5_gs_ho_mc_gsho_homc, full_params5_gs_ho_mc_gsho_homc = build_mnlogit(design_x, y, debug='Y')
test_dev = 2 * (llk5_gs_ho_mc_gsho_homc - llk4_gs_ho_mc_gsho)
test_df = df5_gs_ho_mc_gsho_homc - df4_gs_ho_mc_gsho
test_p_value_gs_ho_mc_gsho_homc = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value_gs_ho_mc_gsho_homc)

# In[29]:


print(f'Degree of Freedom = {df5_gs_ho_mc_gsho_homc}')


# In[30]:


def chi_square_test(x_cat, y_cat, debug='N'):
    obs_count = pandas.crosstab(index=x_cat, columns=y_cat, margins=False, dropna=True)
    c_total = obs_count.sum(axis=1)
    r_total = obs_count.sum(axis=0)
    n_total = numpy.sum(r_total)
    exp_count = numpy.outer(c_total, (r_total / n_total))

    if (debug == 'Y'):
        print('Observed Count:\n', obs_count)
        print('Column Total:\n', c_total)
        print('Row Total:\n', r_total)
        print('Overall Total:\n', n_total)
        print('Expected Count:\n', exp_count)
        print('\n')

    chi_sq_stat = ((obs_count - exp_count) ** 2 / exp_count).to_numpy().sum()
    chi_sq_df = (obs_count.shape[0] - 1.0) * (obs_count.shape[1] - 1.0)
    chi_sq_sig = scipy.stats.chi2.sf(chi_sq_stat, chi_sq_df)

    cramer_v = chi_sq_stat / n_total
    if (c_total.size > r_total.size):
        cramer_v = cramer_v / (r_total.size - 1.0)
    else:
        cramer_v = cramer_v / (c_total.size - 1.0)
    cramer_v = numpy.sqrt(cramer_v)

    return (chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v)


# In[31]:


def deviance_test(x_int, y_cat, debug='N'):
    y = y_cat.astype('category')

    # Model 0 is yCat = Intercept
    x = numpy.where(y_cat.notnull(), 1, 0)
    obj_logit = stats.MNLogit(y, x)
    this_fit = obj_logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    llk0 = obj_logit.loglike(this_parameter.values)

    if (debug == 'Y'):
        print(this_fit.summary())
        print("Model Log-Likelihood Value =", llk0)
        print('\n')

    # Model 1 is yCat = Intercept + xInt
    x = stats.add_constant(x_int, prepend=True)
    obj_logit = stats.MNLogit(y, x)
    this_fit = obj_logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    llk1 = obj_logit.loglike(this_parameter.values)

    if (debug == 'Y'):
        print(this_fit.summary())
        print("Model Log-Likelihood Value =", llk1)

    # Calculate the deviance
    deviance_stat = 2.0 * (llk1 - llk0)
    deviance_df = (len(y.cat.categories) - 1.0)
    deviance_sig = scipy.stats.chi2.sf(deviance_stat, deviance_df)

    mc_fadden_r_sq = 1.0 - (llk1 / llk0)

    return (deviance_stat, deviance_df, deviance_sig, mc_fadden_r_sq)


# In[32]:


cat_pred = ['group_size', 'homeowner', 'married_couple']
int_pred = cat_pred

test_result = pandas.DataFrame(index=cat_pred + int_pred,
                               columns=['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in cat_pred:
    chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v = chi_square_test(purchase_likelihood[pred], purchase_likelihood['insurance'],
                                                                   debug='Y')
    test_result.loc[pred] = ['Chi-square', chi_sq_stat, chi_sq_df, chi_sq_sig, 'Cramer''V', cramer_v]

for pred in int_pred:
    deviance_stat, deviance_df, deviance_sig, mc_fadden_r_sq = deviance_test(purchase_likelihood[pred],
                                                                             purchase_likelihood['insurance'], debug='Y')
    test_result.loc[pred] = ['Deviance', deviance_stat, deviance_df, deviance_sig, 'McFadden''s R^2', mc_fadden_r_sq]

print(test_result[test_result['Test'] == 'Deviance'])

# d)	(5 points) Calculate the Feature Importance Index as the negative base-10 logarithm of the significance value.  List your indices by the model effects.

# In[128]:


print(f'Feature Importance Index for (Intercept + group_size) = {-(numpy.log10(test_p_value_gs))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner) = {-(numpy.log10(test_p_value_gs_ho))}')
print(
    f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple) = {-(numpy.log10(test_p_value_gs_ho_mc))}')
print(
    f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner) = {-(numpy.log10(test_p_value_gs_ho_mc_gsho))}')
print(
    f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple) = {-(numpy.log10(test_p_value_gs_ho_mc_gsho_homc))}')