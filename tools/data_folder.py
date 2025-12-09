import json5

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('data_folder')
class DataFolder(BaseTool):
    # `description` is used to tell the agent the function of the tool.
    description = '存放社交媒体贴文、用户信息等数据的文件夹，输入文件夹名称和数据索引可以取出对应数据，需要注意数据长度不能超过大模型的上下文长度上限。'
    # `parameters` tells the agent what input parameters the tool has.
    parameters = [{
        'name': 'folder_name',
        'type': 'string',
        'description': '文件夹名称',
        'required': True
    },
                  {
        'name': 'start_idx',
        'type': 'int',
        'description': '数据的开始索引（包含）',
        'required': True
    },
                  {
        'name': 'end_idx',
        'type': 'int',
        'description': '数据的结束索引（不包含）',
        'required': True
    },]
    data_folders = {}
    show_funcs = {}
    read_record = {}
    read_count = 0
    
    def initialize(self):
        self.read_count = 0
        self.read_record = {}
        self.data_folders = {}
        self.show_funcs = {} 

    def call(self, params: str, **kwargs):
        """
        Search posts based on location, start time and end time
        :param location: Location
        :param start_time: Start time %Y-%m-%d %H:%M:%S
        :param end_time: End time %Y-%m-%d %H:%M:%S
        :return: Search results list
        """
        params = json5.loads(params)
        folder_name, start_idx, end_idx = params['folder_name'], params['start_idx'], params['end_idx']

        if folder_name not in self.data_folders:
            raise ValueError(f"文件夹'{folder_name}'不存在。请先创建文件夹或检查名称是否正确。")
        
        if folder_name not in self.read_record:
            self.read_record[folder_name] = []
            
        valid_ranges = [[start_idx, end_idx]]
        for si, ei in self.read_record[folder_name]:
            edited = True
            while edited:
                edited = False
                for j, (vsi, vei) in enumerate(valid_ranges):
                    if vsi >= si and ei >= vei:
                        edited = True
                        valid_ranges[j] = [-1, -1]
                        break
                    elif vsi <= si and ei <= vei:
                        edited = True
                        valid_ranges[j] = [-1, -1]
                        valid_ranges.extend([[vsi, si], [ei, vei]])
                        break
                    elif vsi <= si and ei >= vei and si < vei:
                        edited = True
                        valid_ranges[j] = [vsi, si]
                        break
                    elif vsi >= si and ei <= vei and vsi < ei:
                        edited = True
                        valid_ranges[j] = [ei, vei]
                        break
                while [-1, -1] in valid_ranges:
                    valid_ranges.remove([-1, -1])
                    
        if len(valid_ranges) == 0:
            return "***该数据已读取过，请查看历史输出***"
        else:
            result = ''
            for vsi, vei in valid_ranges:
                result += self.show_funcs[folder_name](self.data_folders[folder_name], vsi, vei) + '\n'
                self.read_count += int(vei) - int(vsi)
                self.read_record[folder_name].append([vsi, vei])
            if self.read_count >= 100:
                result += "\n\n***读取的数据过多，请尽快结束推理，避免读取数据。***"
            
            return result
