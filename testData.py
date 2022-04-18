with open("result_back.txt","r") as f:
    lines = f.readlines()

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for line in lines:
    parts = line.split(",")
    parts = [x.strip() for x in parts]
    print(parts[0])
    train_loss_list.append(float(parts[1].split(" : ")[1]))
    train_acc_list.append(float(parts[2].split(" : ")[1]))
    test_loss_list.append(float(parts[3].split(" : ")[1]))
    test_acc_list.append(float(parts[4].split(" : ")[1]))
from matplotlib import pyplot as plt
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