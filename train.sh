python -m train --trainset "/Data/ProtGym/alignments/UBE4B_MOUSE_1_b0.45.a2m" \
			    --valset "/Data/InverseFoldingData/mutational_data/alignments/UBE4B.csv.UBE4B_MOUSE_1_b0.45.a2m.exp" \
			    --save "models/saved_PF00207_PF07677.pth.tar" \
				--load "" \
			    --modelconfig "shallow.config.json" \
				--outputfile "output.txt"
