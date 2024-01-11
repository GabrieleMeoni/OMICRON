import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime



def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=100, device="cpu"):
    
    # plotting = True

    # if plotting == True:
       
    #     # make filename to store figure
    #     timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     filename = './result_plots/training_plot_' + timestr + '.png'

    #     # intialize plot
    #     plt.ion()
    #     plt.style.use('fivethirtyeight')
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.legend([' (red)Training Loss']) #'(green)Validation Loss', '(blue)Accuracy'])
    #     ax.set_ylim([0, 1.5])

    #     # x - axis :array with epochs
    
    # epoch_arr = np.arange(1, epochs+1)

    # # y-axis: storing intermediate results for plotting 
    # cum_train_loss = []
    # cum_val_loss = []
    # acc = [] 


        
    # loop over epochs
    for epoch in range(1, epochs+1):

        # training loop
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            

            # print(inputs.size())
            # inputs = inputs.to(device)
            # targets = targets.to(device)

            # output_edge,output_cloud,output_good = model(inputs)
            # output = torch.cat([output_edge,output_cloud,output_good],dim=1)
            
            # loss = loss_fn(output,targets)
            # loss.backward()

            # optimizer.step()
            # training_loss += loss.data.item() * inputs.size(0)
            
        # training_loss /= len(train_loader.dataset)
            

            for i in range(len(targets)):
                model.cloud.requires_grad = True
                model.edge.requires_grad = True
                model.good.requires_grad = True

                if targets[i]==2: # good
                    print("a")
                    model.cloud.requires_grad = False
                    model.edge.requires_grad = False
                    targeti = torch.tensor([0,0,1])

                elif targets[i] == 0: # cloud
                    print("b")
                    model.edge.requires_grad = False
                    model.good.requires_grad = False
                    targeti = torch.tensor([1,0,0])
                    
                else: # edge
                    print("c")
                    model.cloud.requires_grad = False
                    model.good.requires_grad = False
                    targeti = torch.tensor([0,1,0])
                
              
                output_edge,output_cloud,output_good = model(inputs[i,:,:,:].unsqueeze(0))
                output = torch.cat([output_edge,output_cloud,output_good],dim=0).flatten()
                
                loss = loss_fn(output, targeti.float())
                print("loss", loss)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item()
                print("training_loss",training_loss)

        # training_loss /= len(train_loader.dataset)
        # print('Epoch: {}, Training Loss: {:.2f}'.format(epoch, training_loss))	

        # # validation loop
        # model.eval()
        # num_correct = 0 
        # num_examples = 0
        
        # for batch in val_loader:
        #     inputs, targets = batch
        #     output_edge,output_cloud,output_good = model(inputs)
        #     output = torch.cat([output_edge,output_cloud,output_good],dim=1)

        #     loss = loss_fn(output,targets) 
        #     valid_loss += loss.data.item() * inputs.size(0)
        #     correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
        #     num_correct += torch.sum(correct).item()
        #     num_examples += correct.shape[0]

        # valid_loss /= len(val_loader.dataset)
        
        
        # print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        # valid_loss, num_correct / num_examples))


    #     if plotting == True:
            
    #         # storing intermediate results for plotting
    #         cum_train_loss.append(training_loss)
    #         cum_val_loss.append(valid_loss)
    #         # acc.append(num_correct / num_examples)

    #         # plotting results 
    #         ax.plot(epoch_arr[:epoch], cum_train_loss, 'r',linewidth=1)
    #         ax.plot(epoch_arr[:epoch], cum_val_loss, 'g',linewidth=1)
    #         ax.plot(epoch_arr[:epoch], acc, 'b',linewidth=1)

    #         # refresh figure
    #         ax.legend(['(red)Training Loss', '(green)Validation Loss', '(blue)Accuracy'])
    #         fig.canvas.draw()
    #         fig.canvas.flush_events()

    #         # saving intermediate plots
    #         if epoch % 10 == 0:
    #             fig.savefig(filename)


    # # saving final plot 
    # fig.savefig()

    return