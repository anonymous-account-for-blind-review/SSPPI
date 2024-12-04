import argparse
from my_dataset import collate_pyg, PygDataset
from torch.utils.data import DataLoader
from model.SSPPI import SSPPI
import torch
import pandas as pd
from tqdm import tqdm
import pickle as pkl

def main(args):
    test_pns = []
    test_ori = []
    df1 = pd.read_csv(f'./data/{args.datasetname}/test.csv')
    pro1 = list(df1['proteinA'])
    pro2 = list(df1['proteinB'])
    label = list(df1['label'])
    df2 = pd.read_csv(f'./data/{args.datasetname}/map.csv')

    id_map_dict = dict(zip(list(df2['From']),list(df2['Entry']))) ### Used to convert gene names to protein UniProt IDs.
    for i in range(len(pro1)):
        test_pns.append((id_map_dict[pro1[i]], id_map_dict[pro2[i]], 1))
        test_ori.append((pro1[i],pro2[i]))

    with open(f'./data/{args.datasetname}/index_map_dict.pkl', 'rb') as file:  ### Generated arbitrarily for compatibility with the original architecture, but not actually needed during inference.
        index_map_dict = pkl.load(file)

    test_dataset = PygDataset( dataset_name = args.datasetname, pns=test_pns,map_dict = index_map_dict)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn = collate_pyg)

    device = torch.device('cuda:'+ str(args.device_id) if torch.cuda.is_available() else "cpu")
    model = SSPPI()
    model = model.to(device)
    path = f'./model_pkl/multi_species/model_any.pkl'
    model.load_state_dict(torch.load(path))

    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader)):
            pro_data1 = data[0].to(device)
            pro_data2 = data[1].to(device)
            output,_ = model(pro_data1, pro_data2, device)
            predicted_values = torch.sigmoid(output)
            predicted_labels = torch.round(predicted_values)
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)
            total_true_labels = torch.cat((total_true_labels, pro_data1.y.view(-1, 1).cpu()), 0)
    G, P_value, P_label = total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()
    predict_df = pd.DataFrame({
        'proteinA': pro1,
        'proteinB': pro2,
        'label':label,
        'predicted label': P_label
    })
    predict_df.to_csv(f'./data/{args.datasetname}/predicted_results.csv', index=False)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasetname', type = str , default = 'cancer-onecore', choices=['cancer-onecore', 'cancer-crossover']) ###
    parse.add_argument('--device_id', type = int, default = 1)
    parse.add_argument('--batch_size', type=int, default = 64)
    args = parse.parse_args()

    main(args)