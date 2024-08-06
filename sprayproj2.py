#!/usr/bin/env python
# coding: utf-8

# In[1]:


import trackpy as tp
import pims
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import cv2 as cv
import cv2
from skimage import measure
import feret
from itertools import chain


# In[2]:


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# In[3]:


def binarize(gray_image):
    ret, bin_img = cv2.threshold(gray_image,
                             0, 255, 
                         cv2.THRESH_OTSU)
    return ret, bin_img


# In[4]:


def morphologyex(bin_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)
    return bin_img, kernel


# In[5]:


def morphological_ops(bin_img, kernel):
    #fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # sure background area
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    
    # Distance transform
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    

    #foreground area
    ret, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)  
    

    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    return sure_fg, sure_bg, dist, unknown


# In[6]:


def marker(sure_fg, unknown):
    ret, markers = cv2.connectedComponents(sure_fg)
  
    # Add one to all labels so that background is not 0, but 1
    markers += 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0
  
    
    return ret, markers


# In[7]:


def segmentation(markers, img):
    
    markers = cv2.watershed(img, markers)
    labels = np.unique(markers)

    particles = []
    for label in labels[2:]:  

    # Create a binary image in which only the area of the label is in the foreground 
    #and the rest of the image is in the background   
        target = np.where(markers == label, 255, 0).astype(np.uint8)

      # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        particles.append(contours[0])

    # Draw the outline
    img = cv2.drawContours(img, particles, -1, color=(200, 0, 0), thickness=2)
    
    return markers, img


# In[ ]:





# In[8]:


def particle_properties(region, img):
    
    import math
    aspect_ratio=[]
    length=[]
    width=[]
    x=[]
    y=[]
    i=0
    count=0
    for props in region:
        y0, x0 = props.centroid
        x.append(x0)
        y.append(y0)
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.equivalent_diameter_area
        y1 = y0 - math.sin(orientation) * 0.5 * props.equivalent_diameter_area
        x2 = x0 - math.cos(orientation) * 0.5 * props.equivalent_diameter_area
        y2 = y0 + math.sin(orientation) * 0.5 * props.equivalent_diameter_area


        ar=(props.minor_axis_length/props.major_axis_length)
        aspect_ratio.append(ar)
        length.append(props.feret_diameter_max)
        width.append(props.minor_axis_length)
        #ogdiam.append((props.aspect_ratio))
        #print(props.perimeter)
        count+=1
        if(i==0):
            i+=1
            continue
        #ax.plot((x0, x1), (y0, y1), '-r', linewidth=2)
        #ax.plot((x0, x2), (y0, y2), '-r', linewidth=2)
        #ax.plot(x0, y0, '.g', markersize=1)

        #minr, minc, maxr, maxc = props.bbox
        #bx = (minc, maxc, maxc, minc, minc)
        #by = (minr, minr, maxr, maxr, minr)
        #ax.plot(bx, by, '-b', linewidth=2.5)

    #ax.axis((0, 300, 300, 0))
    
    
    return length, width, x, y, aspect_ratio


# In[9]:


@pims.pipeline
def gray1(image):
    return image[:, :, 1]  # Take just the green channel
frame_rgb=(pims.open(r"C:\Users\Pranav A\Desktop\dptprojs\frames\frames_less\*.jpg"))
frame=gray1(pims.open(r"C:\Users\Pranav A\Desktop\dptprojs\frames\frames_less\*.jpg"))


# In[10]:


frame[0].shape


# In[11]:


def rgbify(frame_img):  
    rgb_frame = np.stack((frame_img,) * 3, axis=-1)
    return rgb_frame


# In[12]:


dict1={}
framelist=[]
lenlist=[]
widlist=[]
xlist=[]
ylist=[]
arlist=[]
for i in range(10):
    #gray=grayscale(frame[i])
    ret, binary=binarize(frame[i])
    binimg, kernel=morphologyex(binary)
    sure_fg, sure_bg, dist, unknown=morphological_ops(binimg, kernel)
    ret, markers=marker(sure_fg, unknown)
    rgb_frame=rgbify(frame[i])
    markers, img=segmentation(markers, rgb_frame)
    plt.imsave(f"marker2/marker_{i}.png", markers)
    #tp.locate(markers, 41, 15, invert=True)
    region=measure.regionprops(markers,intensity_image=img)
    length, width,x, y, ar=particle_properties(region, frame[i])
    for j in range(len(length)):
        framelist.append(i)
    lenlist.append(length)
    widlist.append(width)
    xlist.append(x)
    ylist.append(y)
    arlist.append(ar)
    #ecaddict['i']='diam'


# In[13]:


lenlist=list(chain.from_iterable(lenlist))


# In[14]:


widlist=list(chain.from_iterable(widlist))
xlist=list(chain.from_iterable(xlist))
ylist=list(chain.from_iterable(ylist))
arlist=list(chain.from_iterable(arlist))


# In[15]:


dict1={}
dict1['frame']=framelist
dict1['length']=lenlist
dict1['width']=widlist
dict1['x']=xlist
dict1['y']=ylist
dict1['aspect_ratio']=arlist


# In[16]:


new=pd.DataFrame.from_dict(dict1)


# In[17]:


new


# In[18]:


frame_rgb1=(pims.open(r"C:\Users\Pranav A\Desktop\dptprojs\marker2\*.png"))
frame1=gray1(pims.open(r"C:\Users\Pranav A\Desktop\dptprojs\marker2\*.png"))


# In[20]:


tp.annotate(new[new.frame==(9)], frame1[9])


# In[20]:


f=tp.link(new, 10, memory=3)
f.to_csv("hopethisworks.csv")


# In[21]:


f


# In[22]:


f[f.particle==(5)]


# In[23]:


l=[]
for i in range(max(f.particle)):
    df=f[f.particle==(i)]
    l.append([i, max(df.length), max(df.width), min(df.width), min(df.aspect_ratio)])


# In[24]:


l.pop(0)


# In[25]:


l


# In[26]:


final = pd.DataFrame(l, columns =['Particle_ID', 'Length', 'Width', 'Thickness', 'Aspect_Ratio']) 


# In[27]:


final


# In[28]:


df2=f[f.particle==1]


# In[29]:


needed_df=final[['Length', 'Particle_ID']].copy()


# In[30]:


ndf=final[['Aspect_Ratio', 'Particle_ID']].copy()


# In[31]:


needed_df


# In[32]:


import seaborn as sns
from sklearn.ensemble import IsolationForest


# In[33]:


ndf2=final[['Particle_ID', 'Aspect_Ratio']]


# In[34]:


ndf2


# In[34]:


ndf2.to_csv('asp_ratio.csv')


# In[35]:


import seaborn as sns
sns.boxplot(ndf2.Aspect_Ratio, orient='h')
plt.show()


# In[36]:


ndf2


# In[37]:


from sklearn.model_selection import train_test_split

# Define the stratification column
stratify_col = 'Aspect_Ratio'

# Ensure the stratification column is categorical or binned for stratified sampling
# For continuous data, you may need to bin the data first
ndf2['aspect_ratio_binned'] = pd.qcut(ndf2['Aspect_Ratio'], q=10, duplicates='drop')

# Perform stratified sampling
train_data, validation_data = train_test_split(ndf2, test_size=0.2, stratify=ndf2['aspect_ratio_binned'])

# Drop the temporary binned column
validation_data = validation_data.drop(columns=['aspect_ratio_binned'])
train_data = train_data.drop(columns=['aspect_ratio_binned'])

# Save the validation subset to a new CSV file
validation_data.to_csv('validation_subset.csv', index=False)

print("Validation subset created and saved as 'validation_subset.csv'")


# In[38]:


validation_data


# In[39]:


type(train_data)


# In[40]:


train_data.keys()


# In[41]:


print(train_data['Aspect_Ratio'].isnull().sum())


# In[42]:


import matplotlib.pyplot as plt
plt.boxplot(train_data['Aspect_Ratio'])
plt.xlabel('Aspect Ratio')
plt.show()


# In[43]:


model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=42)

model.fit(train_data[['Aspect_Ratio']])

print(model.get_params())


# In[44]:


train_data['scores'] = model.decision_function(train_data[['Aspect_Ratio']])
train_data['anomaly_score'] = model.predict(train_data[['Aspect_Ratio']])
train_data[train_data['anomaly_score']==-1]


# In[45]:


ndf3=train_data[train_data['anomaly_score']==-1].copy()


# In[46]:


ndf3


# In[47]:


plt.figure(figsize=(5,3))
plt.xlabel("Aspect Ratio")
plt.ylabel("Particle ID")
plt.scatter(train_data["Aspect_Ratio"], train_data["Particle_ID"], color = "b", s = 22)# plot outlier values
plt.scatter(ndf3['Aspect_Ratio'], ndf3["Particle_ID"], color = "r", s=22)


# In[52]:


new_val_ds=pd.read_csv('val.csv')


# In[53]:


new_val_ds


# In[54]:


new_val_ds = new_val_ds.loc[:, ~new_val_ds.columns.str.contains('^Unnamed')]


# In[55]:


new_val_ds


# In[56]:


y=new_val_ds['Class']


# In[57]:


y = y.replace({0: 1, 1: -1})


# In[58]:


y


# In[59]:


x=new_val_ds.drop('Class', 1)


# In[60]:


x


# In[61]:


model.predict(x[['Aspect_Ratio']])


# In[62]:


y_pred=model.predict(x[['Aspect_Ratio']])


# In[63]:


from sklearn.metrics import accuracy_score, f1_score, classification_report


# In[64]:


accuracy_score(y_pred, y)


# In[79]:


f1_score(y, y_pred)


# In[66]:


print(classification_report(y_pred, y))


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


print(confusion_matrix(y_pred, y))


# In[75]:


class_names=['Anomaly','Not Anomaly']


# In[78]:


cm=confusion_matrix(y, y_pred)
fig = plt.figure()
fig,ax=plt.subplots()
cax = ax.imshow(cm)
plt.title('Confusion matrix of the classifier')
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i,j],
                       ha="center", va="center", color="w")

fig.colorbar(cax)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




