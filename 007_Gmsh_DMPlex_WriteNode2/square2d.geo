SetFactory("OpenCASCADE");

// ジオメトリ: 0<=x<=1, 0<=y<=1
Rectangle(1) = {0, 0, 0, 1, 1, 0};

// 物理グループ（後でラベルとして使える）
Physical Surface("domain") = {1};
Physical Curve("boundary") = {1,2,3,4};

// メッシュ設定：2次要素
Mesh.ElementOrder = 2;               // 2次要素
Mesh.SecondOrderIncomplete = 0;      // 完全2次（既定でOK）
Mesh.HighOrderOptimize = 1;          // 幾何最適化（任意）
Mesh.CharacteristicLengthMin = 0.1;  // 粗さは適宜
Mesh.CharacteristicLengthMax = 0.1;
