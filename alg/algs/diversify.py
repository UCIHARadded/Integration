from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import ConcatDataset

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class Diversify(Algorithm):

    def __init__(self, args):
        super(Diversify, self).__init__(args)
        # Feature extractor
        self.featurizer = get_fea(args)
        
        # Domain characterization components
        self.dbottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)
        self.dclassifier = common_network.feat_classifier(
            int(args.latent_domain_num),
            args.bottleneck, 
            args.classifier
        )
        
        # Main classification components
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        
        # Auxiliary classification components
        self.abottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
            int(args.num_classes * args.latent_domain_num),
            args.bottleneck, 
            args.classifier
        )
        
        # Domain discrimination components
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        # Add flag for explainability mode
        self.explain_mode = False  # New flag

    def update_d(self, minibatch, opt):
        """Update domain characterization components"""
        all_x1 = minibatch[0].cuda().float()
        all_d1 = minibatch[1].cuda().long()
        all_c1 = minibatch[4].cuda().long()
        
        # Forward pass
        z1 = self.dbottleneck(self.featurizer(all_x1))
        # Clone output during explainability to prevent inplace modification
        if self.explain_mode:
            z1 = z1.clone()
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        cd1 = self.dclassifier(z1)
        
        # Loss calculation
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')
        ent_loss = Entropylogits(cd1) * self.args.lam + F.cross_entropy(cd1, all_c1)
        loss = ent_loss + disc_loss
        
        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        """Set pseudo-domain labels using clustering"""
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0].cuda().float()
                index = data[-1]
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        
        # Normalize features
        all_output = nn.Softmax(dim=1)(all_output)
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        # Clustering for pseudo-domain labels
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        # Refine clustering
        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        # Handle ConcatDataset
        if isinstance(loader.dataset, ConcatDataset):
            concat_dataset = loader.dataset
            cumulative_sizes = concat_dataset.cumulative_sizes
            datasets = concat_dataset.datasets
            starts = [0] + cumulative_sizes[:-1]
            
            per_ds_indices = [[] for _ in range(len(datasets))]
            per_ds_labels = [[] for _ in range(len(datasets))]
            
            for idx, label in zip(all_index, pred_label):
                for ds_idx, end in enumerate(cumulative_sizes):
                    if idx < end:
                        start = starts[ds_idx]
                        local_idx = idx - start
                        per_ds_indices[ds_idx].append(local_idx)
                        per_ds_labels[ds_idx].append(label)
                        break
            
            for ds_idx, dataset in enumerate(datasets):
                if per_ds_indices[ds_idx]:
                    dataset.set_labels_by_index(
                        per_ds_labels[ds_idx],
                        per_ds_indices[ds_idx],
                        'pdlabel'
                    )
        else:
            loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')
        
        print(Counter(pred_label))
        
        # Return to training mode
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def update(self, data, opt):
        """Update domain-invariant features"""
        all_x = data[0].cuda().float()
        all_y = data[1].cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))
        # Clone output during explainability to prevent inplace modification
        if self.explain_mode:
            all_z = all_z.clone()
        
        # Domain discrimination
        disc_input = Adver_network.ReverseLayerF.apply(all_z, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = data[4].cuda().long()
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        
        # Classification
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        # Combined loss
        loss = classifier_loss + disc_loss
        
        # Optimization step
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        """Update auxiliary classifier"""
        all_x = minibatches[0].cuda().float()
        all_c = minibatches[1].cuda().long()
        all_d = minibatches[4].cuda().long()
        all_y = all_d * self.args.num_classes + all_c
        
        # Forward pass
        all_z = self.abottleneck(self.featurizer(all_x))
        # Clone output during explainability to prevent inplace modification
        if self.explain_mode:
            all_z = all_z.clone()
        all_preds = self.aclassifier(all_z)
        
        # Loss calculation and optimization
        classifier_loss = F.cross_entropy(all_preds, all_y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        
        return {'class': classifier_loss.item()}

    def predict(self, x):
        """Main prediction method"""
        # Clone output during explainability to prevent inplace modification
        features = self.featurizer(x)
        bottleneck_out = self.bottleneck(features)
        if self.explain_mode:
            bottleneck_out = bottleneck_out.clone()
        return self.classifier(bottleneck_out)
    
    def predict1(self, x):
        """Domain discriminator prediction"""
        features = self.featurizer(x)
        bottleneck_out = self.dbottleneck(features)
        if self.explain_mode:
            bottleneck_out = bottleneck_out.clone()
        return self.ddiscriminator(bottleneck_out)
    
    def forward(self, batch):
        """Forward pass with loss calculation"""
        inputs = batch[0]
        labels = batch[1]
        
        # Get predictions
        preds = self.predict(inputs)
        preds = preds.float()
        labels = labels.long()
        
        # Compute classification loss
        class_loss = self.criterion(preds, labels)
        
        return {'class': class_loss}
    
    # New method for SHAP explainability
    def explain(self, x):
        """Safe forward pass for explainability tools"""
        original_mode = self.explain_mode
        try:
            self.explain_mode = True
            with torch.no_grad():
                return self.predict(x)
        finally:
            self.explain_mode = original_mode
