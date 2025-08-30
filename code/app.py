import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import analysis_of_unbalance



st.title("Analysis of Unbalance in Classes")
#st.write("come arroz con leche.")
st.sidebar.title("Menu")

if st.sidebar.checkbox("Plot of intersection classes",):
    #st.sidebar.write("Here are some additional options you can configure.")
    st.write("...")

    if st.sidebar.button("LOS intersection"):
        st.header("LOS - Classes Intersection Analysis")
        info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009, test, report_classes_in_one_of_datasets_s009_less_s008 = analysis_of_unbalance.classes_intersection(type_connection='LOS')

        st.write("Dataset Information:")
        st.write(info_of_dataset)
        st.write("Classes with NaN in s009 less than s008:")
        #df_test_0 = pd.DataFrame(test[0].tolist())
        #print(test[0])
        #class_with_nan_s009_less_s008 = test[0]
        #st.write(df_test_0)
        #st.write("Classes with difference equal to zero in s009 less than s008:")
        #class_with_diff_equal_to_zero_s009_less_s008 = test[1]
        #st.write(class_with_diff_equal_to_zero_s009_less_s008)

        dic_ = {'classes': data_about_intersection_in_s008['classes'],
                'samples_s008': data_about_intersection_in_s008['samples'],
                'samples_s009': data_about_intersection_in_s009['samples']}
        df = pd.DataFrame(dic_)

        st.bar_chart (data=df, x='classes', y=['samples_s008', 'samples_s009'],
                      use_container_width=True, height=400, width=600, y_label='samples')

        dic_ = {'classes': data_about_intersection_in_s008['classes'],
                'samples_s008': data_about_intersection_in_s008['percentage'],
                'samples_s009': data_about_intersection_in_s009['percentage']}
        df = pd.DataFrame (dic_)

        st.bar_chart(data=df, x='classes', y=['samples_s008', 'samples_s009'],
                     use_container_width=True, height=400, width=600, y_label='percentages')

        st.write ("Report of classes in one of the datasets s009 less than s008:")
        # st.write(report_classes_in_one_of_datasets_s009_less_s008)
        df_report = pd.DataFrame (report_classes_in_one_of_datasets_s009_less_s008)
        st.table (df_report)

    if st.sidebar.button ("NLOS intersection"):
        st.header ("NLOS - Classes Intersection Analysis")
        info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009 = analysis_of_unbalance.classes_intersection (
            type_connection='NLOS')

        st.write ("Dataset Information:")
        st.write (info_of_dataset)

        dic_ = {'classes': data_about_intersection_in_s008 ['classes'],
                'samples_s008': data_about_intersection_in_s008 ['samples'],
                'samples_s009': data_about_intersection_in_s009 ['samples']}
        df = pd.DataFrame (dic_)

        st.bar_chart (data=df, x='classes', y=['samples_s008', 'samples_s009'],
                      use_container_width=True, height=400, width=600, y_label='samples')

        dic_ = {'classes': data_about_intersection_in_s008 ['classes'],
                'samples_s008': data_about_intersection_in_s008 ['percentage'],
                'samples_s009': data_about_intersection_in_s009 ['percentage']}
        df = pd.DataFrame (dic_)

        st.bar_chart (data=df, x='classes', y=['samples_s008', 'samples_s009'],
                      use_container_width=True, height=400, width=600, y_label='percentages')

    if st.sidebar.button ("ALL intersection"):
        st.header ("ALL - Classes Intersection Analysis")
        info_of_dataset, data_about_intersection_in_s008, data_about_intersection_in_s009 = analysis_of_unbalance.classes_intersection (
            type_connection='ALL')

        st.write ("Dataset Information:")
        st.write (info_of_dataset)

        dic_ = {'classes': data_about_intersection_in_s008 ['classes'],
                'samples_s008': data_about_intersection_in_s008 ['samples'],
                'samples_s009': data_about_intersection_in_s009 ['samples']}
        df = pd.DataFrame (dic_)

        st.bar_chart (data=df, x='classes', y=['samples_s008', 'samples_s009'],
                      use_container_width=True, height=400, width=600, y_label='samples')

        dic_ = {'classes': data_about_intersection_in_s008 ['classes'],
                'samples_s008': data_about_intersection_in_s008 ['percentage'],
                'samples_s009': data_about_intersection_in_s009 ['percentage']}
        df = pd.DataFrame (dic_)

        st.bar_chart (data=df, x='classes', y=['samples_s008', 'samples_s009'],
                      use_container_width=True, height=400, width=600, y_label='percentages')





    #st.line_chart(intersection_classes_LOS_s008)
    #st.bar_chart(intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s009)
    #fig, ax = plt.subplot(1, 1, 1)
    #ax.bar((intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s008), label='s008')
    #ax.bar((intersection_classes_LOS_s008, sample_of_intersection_classes_LOS_s009), label='s009')
    #st.pyplot(fig)
    #st.image ("../results/accuracy/1_headmap_coord_in_termometro_s008_s008.png", caption="Heatmap Example")

st.button("Click Me!")
st.slider("Select a value", 0, 100, 50)
st.text_input("Enter some text", "Type here...")