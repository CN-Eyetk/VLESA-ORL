# ESCONV
# EmoSp
CBS Polyu Project for ESCONV

925-I added loss on reponse emotion
925-II added MISC Performance
925-III change batch size to 20

Result 9-27

|         | misc_      | our_wotrans_woppd | our_woppd  | our        |
|---------|------------|-------------------|------------|------------|
| acc     | 0.325      | 0.319             | 0.324      | **0.332**  |
| ppl     | 16.282     | 15.885            | 15.880     | **15.573** |
| length  | 16.624     | 16.037            | **16.904** | 16.126     |
| dist-1  | 3.988      | 4.083             | 4.035      | **4.246**  |
| dist-2  | 17.235     | 17.541            | 17.5       | **18.063** |
| dist-3  | 31.501     | 31.995            | 31.791     | **32.665** |
| bleu-1  | **17.361** | 16.735            | 17.25      | 16.601     |
| bleu-2  | **7.079**  | 6.831             | 7.022      | 6.615      |
| bleu-3  | **3.607**  | 3.456             | 3.545      | 3.366      |
| bleu-4  | **2.095**  | 1.97              | 2.012      | 1.91       |
| f1      | **20.938** | 20.902            | 20.581     | 19.835     |
| rouge-l | 17.591     | **17.618**        | 17.171     | 16.521     |

Comparison with State of the Arts
|       | MISC        | TransESC    | KEMI        | GLHG        | MultiESC | Our         |
|-------|-------------|-------------|-------------|-------------|----------|-------------|
| Event | ACL22       | ACL23       | ACL23       | IJCAI-22    | EMNLP-22 |             |
| LM    | Blender-Bot | Blender-Bot | Blender-Bot | Blender-Bot | Bart     | Blender-Bot |
| acc   | 31.63       | 34.71       | -           | -           | 38~42    | 33.2        |
| ppl   | 16.16       | 15.85       | 15.92       | 15.67       | 15.41    | 15.57       |