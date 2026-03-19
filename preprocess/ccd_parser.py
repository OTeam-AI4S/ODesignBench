import os
import json
import gemmi
from typing import Optional, Dict
from rdkit import Chem

class LocalCcdParser:
    """
    一个用于从本地 components.cif 文件中解析化学成分信息的类。

    该类会自动为 components.cif 文件创建并缓存一个索引，以实现快速查找。
    首次实例化时，如果索引文件不存在，会花费一些时间来构建索引。
    后续实例化将直接加载缓存的索引，速度非常快。
    """
    # 特殊配体的硬编码修复字典（处理 CCD 中格式有缺陷的常见配体）
    FIXED_SMILES = {
        'FMC': 'Nc1ncnc2c1[nH]nc2[C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O',  # Formycin (CACTVS canonical version)
        # 可以在这里添加更多需要修复的配体
    }
    
    # 金属离子的 SMILES 表示
    METAL_IONS = {
        'ZN': '[Zn+2]',
        'MG': '[Mg+2]',
        'CA': '[Ca+2]',
        'FE': '[Fe+2]',
        'FE2': '[Fe+2]',
        'FE3': '[Fe+3]',
        'MN': '[Mn+2]',
        'CU': '[Cu+2]',
        'NA': '[Na+]',
        'K': '[K+]',
    }
    
    def __init__(self, cif_path: str):
        if not os.path.exists(cif_path):
            raise FileNotFoundError(f"指定的 CIF 文件不存在: {cif_path}")
        
        self.cif_path = cif_path
        self.index_path = f"{cif_path}.index.json"
        self.index: Dict[str, int] = self._load_or_build_index()

    def _build_index(self) -> Dict[str, int]:
        """
        (此函数无需修改，本身就是正确的)
        遍历 CIF 文件，构建 CCD code 到其在文件中字节偏移量的索引。
        """
        print(f"索引文件 '{self.index_path}' 不存在，正在构建索引... (这可能需要一两分钟)")
        index = {}
        with open(self.cif_path, 'rb') as f:
            offset = 0
            for line_bytes in f:
                # 仅在需要检查内容时解码
                if line_bytes.startswith(b'data_'):
                    line_str = line_bytes.decode('utf-8', errors='ignore').strip()
                    code = line_str[5:]
                    index[code] = offset
                offset = f.tell()
        
        with open(self.index_path, 'w') as f_out:
            json.dump(index, f_out)
        
        print("索引构建完成并已保存。")
        return index

    def _load_or_build_index(self) -> Dict[str, int]:
        if os.path.exists(self.index_path):
            print(f"正在从 '{self.index_path}' 加载已缓存的索引...")
            with open(self.index_path, 'r') as f:
                return json.load(f)
        else:
            return self._build_index()

    def get_smiles(self, ccd_code: str) -> Optional[str]:
        """
        (*** 已修正：添加 RDKit 验证和标准化 ***)
        从本地 CIF 文件中获取给定 CCD code 的 SMILES 字符串。
        返回的 SMILES 经过 RDKit 验证，确保 Chai1 能够正确解析。
        """
        code = ccd_code.upper()
        
        # 1. 检查是否为金属离子
        if code in self.METAL_IONS:
            smiles = self.METAL_IONS[code]
            # 验证金属离子 SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return smiles
            else:
                print(f"警告：金属离子 '{code}' 的 SMILES '{smiles}' 无效")
                return None
        
        # 2. 检查是否有硬编码修复
        if code in self.FIXED_SMILES:
            smiles = self.FIXED_SMILES[code]
            # 验证修复后的 SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, isomericSmiles=True)
            else:
                print(f"警告：硬编码修复的 SMILES 对于 '{code}' 无效，尝试从 CIF 读取")
        
        # 3. 从 CIF 文件读取
        if code not in self.index:
            print(f"错误：在索引中未找到 CCD code '{code}'。")
            return None

        start_offset = self.index[code]
        
        # *** 修正点 1: 必须使用二进制模式 'rb' 打开文件 ***
        with open(self.cif_path, 'rb') as f:
            f.seek(start_offset)
            
            # 读取该 code 对应的数据块（字节串形式）
            cif_block_bytes_list = []
            for line_bytes in f:
                # *** 修正点 2: 对字节串进行判断 ***
                # 遇到下一个数据块的开头就停止
                if line_bytes.startswith(b'data_') and cif_block_bytes_list:
                    break
                cif_block_bytes_list.append(line_bytes)
            
            # 将字节串列表连接起来，然后一次性解码为字符串
            cif_block_bstring = b"".join(cif_block_bytes_list)
            cif_block_str = cif_block_bstring.decode('utf-8', errors='ignore')

        # 使用 Gemmi 解析这个小的数据块字符串
        raw_smiles = None
        try:
            doc = gemmi.cif.read_string(cif_block_str)
            block = doc.sole_block()
            
            smiles_loop = block.find_loop('_pdbx_chem_comp_descriptor.descriptor')
            if smiles_loop is None:
                return None
            
            # 查找SMILES类型的描述符
            # 通常格式是: descriptor_type, descriptor
            raw_smiles = None
            try:
                # 查找所有SMILES相关的描述符
                type_loop = block.find_loop('_pdbx_chem_comp_descriptor.type')
                descriptor_loop = block.find_loop('_pdbx_chem_comp_descriptor.descriptor')
                
                if type_loop is not None and descriptor_loop is not None:
                    # 将循环转换为列表以便索引
                    type_list = list(type_loop)
                    descriptor_list = list(descriptor_loop)
                    
                    # 遍历查找类型为 'SMILES' 的描述符
                    for i, desc_type in enumerate(type_list):
                        if desc_type and 'SMILES' in str(desc_type).upper():
                            if i < len(descriptor_list):
                                raw_smiles = str(descriptor_list[i]).strip('"\'')
                                break
            except Exception as e:
                # 如果查找失败，继续尝试其他方法
                pass
            
            # 如果找不到，尝试从循环中获取第一个SMILES值
            if raw_smiles is None:
                if isinstance(smiles_loop, list) and len(smiles_loop) > 0:
                    # 如果返回的是列表，取第一个
                    raw_smiles = str(smiles_loop[0]).strip('"\'')
                elif hasattr(smiles_loop, '__iter__') and not isinstance(smiles_loop, str):
                    # 如果是可迭代对象但不是字符串，取第一个元素
                    raw_smiles = str(next(iter(smiles_loop))).strip('"\'')
                else:
                    raw_smiles = str(smiles_loop).strip('"\'')
            
            if raw_smiles is None or not raw_smiles:
                return None

        except Exception as e:
            print(f"解析 CCD code '{code}' 时出错: {e}")
            return None
        
        # 4. 强制验证与标准化（关键改进）
        # 使用 RDKit 验证和标准化 SMILES
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            # 尝试修复：如果 RDKit 无法解析，尝试开启非严格模式
            mol = Chem.MolFromSmiles(raw_smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    print(f"警告：CCD code '{code}' 的 SMILES '{raw_smiles}' 无法通过 RDKit 验证，已丢弃")
                    return None  # 依然失败则丢弃，避免引发 Chai1 崩溃
            else:
                print(f"警告：CCD code '{code}' 的 SMILES '{raw_smiles}' 无法被 RDKit 解析，已丢弃")
                return None
        
        # 重新生成 canonical SMILES，确保 Chai1 的 RDKit 环境能识别
        try:
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return canonical_smiles
        except Exception as e:
            print(f"警告：生成 canonical SMILES 时出错 (code='{code}'): {e}")
            return None