train = pd.read_csv('C:/train/train.csv')
train.head()
train_img = []
for img_name in tqdm(train['id']):

# defining the image path
impath = 'C:/train/'+str(img_name)+'.png'

# reading the image
img = imread
(impath, as_gray=True)

# normalizing the pixel values
img /= 255.0

# converting the type of pixel to float
img = img.astype('float32')

# appending the image into the list
train_img.append
(img
)
# converting the list to numpy array
train_x = np.array(train_img)

# defining the target
train_y = train['label'].values
print(train_x.shape)
# print >>> (60000, 28, 28)