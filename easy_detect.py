# import cv2
# import numpy as np
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_coords
# def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
#     if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text(position, text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
# if __name__ == "__main__":
#     tiny_dict = {0: "烂茧", 1: "黄茧", 2: "伤茧", 3: "拆茧"}  # 分类
#     color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 颜色
#     cap = cv2.VideoCapture(0)
#     weights="runs/train/exp12/weights/best.pt" #在这里修改模型地址
#     w = str(weights[0] if isinstance(weights, list) else weights)
#     model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location='cpu')   #加载模型
#     height, width = 640, 640
#
#     while (1):
#         ret,frame = cap.read()
#         #frame = cv2.imread('data/cocoons/images/strengthen/112be.jpg')  # 在这里选择测试图片
#         img = cv2.resize(frame, (height, width))  # 尺寸变换
#         img = img / 255.
#         img = img[:, :, ::-1].transpose((2, 0, 1))  # HWC转CHW
#         img = np.expand_dims(img, axis=0)  # 扩展维度至[1,3,640,640]
#         img = torch.from_numpy(img.copy())  # numpy转tensor
#         img = img.to(torch.float32)  # float64转换float32
#         pred = model(img, augment='store_true', visualize='store_true')[0]
#         pred.clone().detach()
#         pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)  # 非极大值抑制
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         # 图像显示
#         for i, det in enumerate(pred):
#             if len(det):
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
#                 for *xyxy, conf, cls in reversed(det):
#                     print('{},{},{}'.format(xyxy, conf.numpy(), cls.numpy()))  # 输出结果：xyxy检测框左上角和右下角坐标，conf置信度，cls分类结果
#                     frame = cv2.rectangle(frame, (int(xyxy[0].numpy()), int(xyxy[1].numpy())),
#                                           (int(xyxy[2].numpy()), int(xyxy[3].numpy())), color_list[int(cls.numpy())], 2)
#                     frame = cv2AddChineseText(frame, tiny_dict[int(cls.numpy())],
#                                               (int(xyxy[0].numpy()), int(xyxy[1].numpy()) - 30),
#                                               color_list[int(cls.numpy())], 30)
#
#         #cv2.imwrite('out.jpg', frame)  # 简单画个框
#         cv2.imshow('draw', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     # 释放对象和销毁窗口
#     cap.release()
#     cv2.destroyAllWindows()