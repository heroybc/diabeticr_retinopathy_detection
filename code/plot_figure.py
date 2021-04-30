
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform = transforms.Compose(
    [transforms.Resize(size=(380,380)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



class kfdeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0],)
        img_name += '.jpeg'
        image = Image.open(img_name)
        if self.transform:
             image= self.transform(image)
        landmarks = self.landmarks_frame.iloc[idx, 1]

        sample = {'image': image, 'landmarks': landmarks}

        return sample





    torch.save(model.state_dict(), args.save_modeldir + '/' +  str(epoch) + '_' + str(accuracy) + '.pth')

    print('Accuracy of the network on the test images:%.4f'%accuracy)
    f = open("./record/history" + str(epoch) + ".txt", "a")
    f.write("精准率=" + str(accuracy_score(y_true, y_pred)) + "\n")
    f.write("召回率=" + str(recall_score(y_true, y_pred, average=None)) + "\n")
    f.write("精确率" + str(precision_score(y_true, y_pred, average=None)) + "\n")
    f.write("kappa=" + str(cohen_kappa_score(y_true, y_pred)) + "\n")
    f.write("F1=" + str(f1_score(y_true, y_pred, average=None)) + "\n")
    f.close()

    y_true = label_binarize(y_true, classes=[i for i in range(nb_classes)])
    y_pred = label_binarize(y_pred, classes=[i for i in range(nb_classes)])


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig("./record/ROC_"+str(epoch)+".png")
    # plt.show()


print('Finished Training')
