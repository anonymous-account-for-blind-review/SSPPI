# activate virtual env
#conda activate SSPPI
#cd source directory
cd /media/SSPPI

echo "Start Inferencing ..."
echo "##################################################################################################"
echo "\n"

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Yeast dataset"
python inference.py --output_dim 1 --datasetname yeast --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under 'any' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity any --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under '01' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity 01 --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under '10' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity 10 --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under '25' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity 25 --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under '40' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity 40 --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under 'cold start s1' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity s2 --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-species dataset under 'cold start s2' condition"
python inference.py --datasetname multi_species --output_dim 1 --identity s2 --device_id 1
wait

echo "********************************************************"
echo "Test the performance of the SSPPI model on the Multi-class dataset"
python inference.py --datasetname multi_class --output_dim 7  
wait

echo "********************************************************"
echo "\n"

echo "##################################################################################################"
echo "END ..."
