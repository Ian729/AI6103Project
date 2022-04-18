from matplotlib import pyplot as plt
train_loss_list = [1.4416, 0.9453, 0.7307, 0.6102, 0.5271, 0.4782, 0.4295, 0.3938, 0.3577, 0.3357]

train_acc_list = [46.99, 66.46, 74.38, 78.62, 81.63, 83.40, 85.05, 86.22, 87.61, 88.41]

test_loss_list = [1.4689, 0.8751, 0.8912, 0.6142, 0.5431, 0.4962, 0.4815, 0.5569, 0.4595, 0.4675]

test_acc_list = [49.61, 69.56, 70.49, 79.51, 82.10, 82.95, 83.70, 81.34, 84.69, 84.18]

plt.title("Train Loss vs Test Loss")
plt.plot(train_loss_list, color="blue", label="train loss")
plt.plot(test_loss_list, color="orange", label="test loss")
plt.legend()
plt.savefig("loss.png")
plt.clf()
plt.title("Train Accuracy vs Test Accuracy")
plt.plot(train_acc_list, color="blue", label="train accuracy")
plt.plot(test_acc_list, color="orange", label="test accuracy")
plt.legend()
plt.savefig("accuracy.png")