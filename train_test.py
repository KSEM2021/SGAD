from utils import test_eval,compute_pos_weight,weighted_binary_cross_entropy,reconstruction_loss
from model import *
import os
import copy
import numpy as np

    

class SGADetector:
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.model = SGAD(**model_config).to(train_config['device'])

    def train(self):
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])


        # =====================================================
        # Training
        # =====================================================
        for e in range(self.train_config['epochs']):  
            self.model.train()      
            epoch_loss = 0.0
            # 每个 epoch 开始时清空梯度
            optimizer.zero_grad()
            num_train_datasets = len(self.data['train'])
            for didx, train_data in enumerate(self.data['train']):
                
                train_graph = self.data['train'][didx].graph
                train_graph = train_graph.to(self.train_config['device'])
                output = self.model(train_graph)
                label = train_graph.ano_labels.to(self.train_config['device'])
                homophily = train_graph.homophily
                high = homophily >= self.model_config['high_homophily_threshold']
                low = (homophily >= 0.0) & (homophily < self.model_config['low_homophily_threshold'])
        
                normal = (label == 0)
                anomaly = (label == 1) 

                high_normal =  high & normal
                low_anomaly = low & anomaly
                valid = high_normal + low_anomaly

                low_normal = valid & low & normal
                high_anomaly = valid & high & anomaly

                valid_ni = low_normal + high_anomaly
    
                
                loss_cons = self.model.get_train_loss(train_graph, output['latent'], output['output'], label)
                #_, sim_score =  self.model.get_test_score(train_graph, output['latent'], output['output'])
                _, score_proto = self.model.get_test_score(train_graph, output['raw'], output['latent'], output['output'],self.model_config['beta'])
                #loss = self.model.cross_attn.get_train_loss(residual_embed, train_graph.ano_labels,
                #                                            self.model_config['num_prompt'])
                loss_bce = weighted_binary_cross_entropy(score_proto[valid], label[valid], pos_weight=compute_pos_weight(train_graph.ano_labels).to(self.train_config['device']))
                loss_bce2 = weighted_binary_cross_entropy((1-score_proto)[valid_ni], label[valid_ni], pos_weight=compute_pos_weight(train_graph.ano_labels).to(self.train_config['device']))
                
                #loss_bce2 = weighted_binary_cross_entropy(output['sharpe_score'], label, pos_weight=compute_pos_weight(train_graph.ano_labels).to(self.train_config['device']))
                #loss_score = F.binary_cross_entropy_with_logits(score_proto,label.float())
                #loss_rec = gala_loss(output["raw"], output["latent"], output['output'], normal)
                #loss_rec = rec_loss(output["raw"], output['output'], normal)
                #loss =  loss_cons # + loss_rec #+ 0.05*loss_rec # + loss_bce #+ 0.2*loss_bce1 +0.1*loss_bce2
                #loss = spectral_anomaly_loss(all_cls_score,label)
                loss = loss_cons + loss_bce + loss_bce2#+ 0.1*loss_rec #+ loss_bce
                #optimizer.zero_grad()
                #loss.backward()
                #optimizer.step()
                
                epoch_loss +=loss.item()
                (loss / num_train_datasets).backward()

                # 这里只用于记录 loss，不参与反向传播
                #epoch_loss += loss.item()

            # =====================================================
            # 所有训练数据集的梯度累积完成后统一更新
            # =====================================================
            optimizer.step()
            epoch_loss /= num_train_datasets
            
            #optimizer.zero_grad()
            #epoch_loss.backward()
            #optimizer.step()

        
        '''
        # =====================================================
        # Save checkpoint
        # =====================================================
        checkpoint_dir = "./checkpoint_new2"
        os.makedirs(checkpoint_dir, exist_ok=True)

        seed = self.train_config.get("seed", 0)
        epoch = self.train_config.get("epochs")
        lr = self.model_config.get('lr')

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"SGAD_seed{seed}_epoch{epoch}_lr{lr}.pt"
        )

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": self.train_config["epochs"],
            "model_config": self.model_config,
            "train_config": self.train_config,
        }, checkpoint_path)

        print(f"Model saved to {checkpoint_path}")
        '''

        test_score_list = {}

        self.model.eval()
        with torch.no_grad():
            for didx, test_data in enumerate(self.data['test']):
                test_graph = test_data.graph
                labels = test_graph.ano_labels.to(self.train_config['device'])
                test_graph = test_graph.to(self.train_config['device'])             
                output = self.model(test_graph)
                print(self.train_config['testdsets'][didx])
        
                score = output['score'] 

                test_score = test_eval(labels, score)

                test_data_name = self.train_config['testdsets'][didx]
                #plot_similarity_distribution(sim,test_data_name)
                test_score_list[test_data_name] = {
                    'AUROC': test_score['AUROC'],
                    'AUPRC': test_score['AUPRC'],
                    
                }
            return test_score_list
