import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def acc_overall(test_result, testgold):

  correct = 0
  total = 0
  # count number of correctly diacritized words
  for i in range(len(testgold)):
    for m in range(len(testgold[i].split())):
      if test_result[i].split()[m] == testgold[i].split()[m]:
        correct += 1
      total +=1

  return correct / total

