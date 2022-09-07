# Hướng dẫn bạn sử dụng YOLOV5 để đào tạo mô hình phát hiện mục tiêu của riêng bạn

Xin chào mọi người. Mình là Kiên. Nó đã không được cập nhật trong vài tháng. 
Tôi đã kiểm tra các mục sau đây trong hai ngày qua và đột nhiên có hơn 1 nghìn 
bạn bè theo dõi. Họ phải là bạn trong loạt bài hướng dẫn về bài tập lớn. Vì có 
rất nhiều bạn đang chú ý đến bộ bài tập lớn này, và sắp đến ngày hoàn thành đồ 
án và nộp bài tập lớn rồi nên mình mới cập nhật lại. So với vấn đề phân loại rau 
quả và nhận dạng rác trước đây, nội dung của số này đã được nâng cấp về mặt nội 
dung và mới lạ hơn. Lần này, chúng tôi sẽ sử dụng YOLOV5 để đào tạo một mô hình 
phát hiện mặt nạ, phù hợp hơn với tình hình dịch bệnh hiện nay, và mục tiêu Ngoài 
ra làm bài tập lớn cho mọi người, nội dung này còn có thể làm đồ án tốt nghiệp cho 
một số đối tác nhỏ. Không cần quảng cáo thêm, chúng ta hãy bắt đầu nội dung hôm nay.

Trước tiên, hãy xem hiệu ứng mà chúng tôi muốn đạt được. Chúng tôi sẽ đào tạo mô hình
phát hiện mặt nạ thông qua dữ liệu và đóng gói nó với pyqt5 để thực hiện các chức năng 
phát hiện mặt nạ hình ảnh, phát hiện mặt nạ video và phát hiện mặt nạ thời gian thực 
của máy ảnh.
![image-20211212181048969](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212181048969.png)

![image-20211212194124635](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212194124635.png)

## Tải xuống mã

Địa chỉ tải xuống của mã là：[[YOLOV5-mask-42: Hệ thống phát hiện mặt nạ dựa trên video dạy học do 
YOLOV5 cung cấp (gitee.com)](https://gitee.com/song-laogou/yolov5-mask-42)](https://github.com/ultralytics/yolov5)

![image-20211214191424378](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211214191424378.png)

## Môi trường cấu hình

Đối với các bạn anaconda chưa quen với pycharm, vui lòng đọc blog csdn này trước để hiểu các hoạt động cơ bản của pycharm và anaconda

[Cách định cấu hình môi trường ảo của anaconda trong blog của pycharm_dejahu - CSDN blog_how để định cấu hình anaconda trong pycharm](https://blog.csdn.net/ECHOSON/article/details/117220445)

Sau khi cài đặt xong anaconda, vui lòng chuyển sang nguồn trong nước để tăng tốc độ tải, lệnh như sau:

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```



Đầu tiên tạo một môi trường ảo cho python 3.9, vui lòng thực hiện các thao tác sau trên dòng lệnh：

```bash
conda create -n yolo5 python==3.9
conda activate yolo5
```

### cài đặt pytorch (cài đặt phiên bản gpu và phiên bản cpu)

Tình hình thử nghiệm thực tế là YOLOv5 dùng được trong điều kiện cả CPU và GPU, nhưng tốc độ luyện trong điều kiện CPU sẽ quá khủng, nên các bạn có điều kiện thì phải cài phiên bản GPU Pytorch, còn các bạn không có điều kiện thì nhất quyết phải thuê một máy chủ để sử dụng.
Để biết các bước cài đặt phiên bản GPU cụ thể, vui lòng tham khảo bài viết này：[Cài đặt phiên bản GPU của Tensorflow và Pytorch trong Windows vào năm 2021](https://blog.csdn.net/ECHOSON/article/details/118420968)

Cần lưu ý những điểm sau：

* Trước khi cài đặt, hãy đảm bảo cập nhật trình điều khiển cạc đồ họa của bạn, truy cập trang web chính thức để tải xuống cài đặt trình điều khiển mô hình tương ứng
* Các card đồ họa dòng 30 chỉ có thể sử dụng phiên bản cuda11
* Đảm bảo tạo một môi trường ảo để không có xung đột giữa các khuôn khổ học sâu khác nhau

Những gì tôi đã tạo ở đây là môi trường python3.8, phiên bản đã cài đặt của Pytorch là 1.8.0 và lệnh như sau:

```cmd
    conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # Lưu ý rằng lệnh này chỉ định phiên bản của Pytorch và phiên bản của cuda
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # Đối tác nhỏ của CPU có thể trực tiếp thực hiện lệnh này
```

Sau khi cài đặt xong, hãy kiểm tra xem GPU có

![image-20210726172454406](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210726172454406.png)

### Cài đặt pycocotools

<font color='red'>Sau đó, tôi đã tìm thấy một phương pháp cài đặt đơn giản hơn trong Windows. Bạn có thể sử dụng lệnh sau để cài đặt trực tiếp mà không cần tải xuống rồi cài đặt</font>

```
pip install pycocotools-windows
```

### Cài đặt các gói khác

Ngoài ra, bạn cũng cần cài đặt các gói khác theo yêu cầu của chương trình, bao gồm opencv, matplotlib và các gói này, tuy nhiên việc cài đặt các gói này tương đối đơn giản, có thể thực hiện trực tiếp thông qua lệnh pip. Chúng tôi cd vào thư mục của mã yolov5 và thực hiện trực tiếp các lệnh sau: Quá trình cài đặt gói có thể hoàn tất.
```bash
pip install -r requirements.txt
pip install pyqt5
pip install labelme
```

### bài kiểm tra

Thực thi đoạn mã sau trong thư mục yolov5

```bash
python detect.py --source data/images/bus.jpg --weights pretrained/yolov5s.pt
```

Sau khi thực hiện, thông tin sau sẽ được xuất ra

![image-20210610111308496](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610111308496.png)

Kết quả sau khi phát hiện có thể được tìm thấy trong thư mục running

![image-20210610111426144](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610111426144.png)

Theo hướng dẫn chính thức, mã phát hiện ở đây rất mạnh và hỗ trợ phát hiện nhiều loại hình ảnh và video, cách sử dụng cụ thể như sau：

```bash
 python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```



## xử lí dữ liệu

Điều này được thay đổi thành biểu mẫu chú thích của yolo, và sau đó nội dung của giai đoạn đầu tiên của quá trình chuyển đổi dữ liệu sẽ được xuất bản đặc biệt。

Ghi nhãn dữ liệu Phần mềm được đề xuất ở đây là labelimg, có thể được cài đặt thông qua lệnh pip

Thực thi lệnh `pip install labelimg -i https://mirror.baidu.com/pypi/simple` trong môi trường ảo của bạn để cài đặt, sau đó thực thi trực tiếp phần mềm labelimg trên dòng lệnh để khởi động phần mềm ghi nhãn dữ liệu.


![image-20210609172156067](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210609172156067.png)

Giao diện sau khi khởi động phần mềm như sau：

![image-20210609172557286](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210609172557286.png)

### Chú thích dữ liệu

Mặc dù là đào tạo theo mô hình của yolo nhưng ở đây chúng tôi vẫn chọn cách gán nhãn ở định dạng voc, thứ nhất để sử dụng bộ dữ liệu ở các mã khác rất tiện lợi, thứ hai tôi cung cấp cách chuyển đổi định dạng dữ liệu.
**Quy trình dán nhãn là：**

**1.Mở thư mục hình ảnh**

![image-20210610004158135](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610004158135.png)

**2.Đặt thư mục lưu các tệp chú thích và thiết lập lưu tự động**

![image-20210610004215206](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610004215206.png)

**3.Bắt đầu gắn nhãn, đóng khung, gắn nhãn nhãn của mục tiêu, `crtl + s` để lưu, sau đó d chuyển sang nhãn tiếp theo để tiếp tục gắn nhãn, lặp lại**

![image-20211212201302682](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212201302682.png)

Các phím tắt của labelimg như sau. Học các phím tắt có thể giúp bạn cải thiện hiệu quả của việc gắn nhãn dữ liệu。

![image-20210609171855504](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210609171855504.png)

Sau khi chú thích xong, bạn sẽ nhận được một loạt các tệp txt. Txt ở đây là tệp chú thích của phát hiện mục tiêu. Tên của tệp txt và tệp hình ảnh tương ứng 1-1, như được hiển thị trong hình sau:

![image-20211212170509714](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212170509714.png)

Mở tệp chú thích cụ thể và bạn sẽ thấy nội dung sau. Mỗi dòng trong tệp txt đại diện cho một mục tiêu, được phân biệt bằng dấu cách và đại diện cho id danh mục của mục tiêu và tọa độ trung tâm được chuẩn hóa x, tọa độ y, w và h của hộp đích.

![image-20211212170853677](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212170853677.png)

**4.Sửa đổi tệp cấu hình tập dữ liệu**

Vui lòng đặt dữ liệu đã đánh dấu theo định dạng sau, thuận tiện cho việc lập chỉ mục của chương trình。

```bash
YOLO_Mask
└─ score
       ├─ images
       │    ├─ test # Dưới đây là hình ảnh bộ thử nghiệm
       │    ├─ train # Dưới đây là hình ảnh của tập huấn
       │    └─ val # Đặt hình ảnh bộ xác thực bên dưới
       └─ labels
              ├─ test # Đặt nhãn bộ thử nghiệm bên dưới
              ├─ train # Đặt nhãn bộ đào tạo bên dưới
              ├─ val # Đặt nhãn bộ xác thực bên dưới
```

Tệp cấu hình ở đây là để thuận tiện cho quá trình đào tạo sau này của chúng tôi. Chúng tôi cần tạo tệp `mask_data.yaml` trong thư mục dữ liệu, như thể hiện trong hình sau:

![image-20211212174510070](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212174510070.png)

Đến đây, phần xử lý tập dữ liệu về cơ bản đã xong, và nội dung tiếp theo sẽ là huấn luyện mô hình!

## Đào tạo mẫu

### Đào tạo cơ bản về mô hình

Tạo tệp cấu hình mô hình `mask_yolov5s.yaml` dưới các mô hình, nội dung như sau:

![image-20211212174749558](C:\Users\chenmingsong\AppData\Roaming\Typora\typora-user-images\image-20211212174749558.png)

Trước khi đào tạo mô hình, hãy đảm bảo rằng các tệp sau đây nằm trong thư mục mã

![image-20211212174920551](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212174920551.png)

Thực thi đoạn mã sau để chạy chương trình:

```
python train.py --data mask_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 100 --batch-size 4 --device cpu
```

![image-20210610113348751](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610113348751.png)

Sau khi mã huấn luyện được thực thi thành công, thông tin sau sẽ được xuất ra trên dòng lệnh, bước tiếp theo là bạn yên tâm chờ quá trình huấn luyện mô hình kết thúc。

![image-20210610112655726](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610112655726.png)

Theo kich thuoc cua dat va kiem soat, mo hinh duoc chup tu sau mot thoi gian dai, cong suat phat trien.：

![image-20210610134412258](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610134412258.png)

Mô hình được đào tạo và tệp nhật ký có thể được tìm thấy trong thư mục `train/running/exp3`

![image-20210610145140340](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typo
Tất nhiên, có một số thao tác phức tạp, ví dụ: bạn có thể tiếp tục đào tạo mô hình được nửa chừng từ điểm gián đoạn. Các thao tác này sẽ để bạn tự khám phá.。

## Đánh giá mô hình

Ngoài hiệu ứng phát hiện mà bạn có thể thấy ở phần đầu của blog, cũng có một số chỉ số đánh giá học tập được sử dụng để thể hiện hiệu suất của mô hình của chúng tôi. Chỉ số đánh giá được sử dụng phổ biến nhất để phát hiện mục tiêu là mAP, nằm trong khoảng từ 0 đến A số giữa 1, số càng gần 1 thì hiệu suất mô hình của bạn càng tốt.

Nói chung, chúng ta sẽ tiếp xúc với hai chỉ số, đó là độ thu hồi và độ chính xác. Hai chỉ số p và r chỉ đơn giản là để đánh giá chất lượng của mô hình từ một góc độ và chúng đều có giá trị từ 0 đến 1. Trong số đó, gần bằng 1 có nghĩa là hiệu suất của mô hình tốt hơn và gần bằng 0 có nghĩa là hiệu suất của mô hình kém hơn. Để đánh giá toàn diện hiệu suất của phát hiện mục tiêu, bản đồ mật độ trung bình thường được sử dụng để đánh giá thêm chất lượng của mô hình. Bằng cách đặt các ngưỡng tin cậy khác nhau, chúng ta có thể nhận được giá trị p và giá trị r được mô hình tính toán theo các ngưỡng khác nhau. Nói chung, giá trị p và giá trị r có tương quan nghịch và chúng có thể được vẽ như hình bên dưới. Như được hiển thị trong đường cong, khu vực của đường cong được gọi là AP. Mỗi mục tiêu trong mô hình phát hiện mục tiêu có thể tính toán một giá trị AP và giá trị mAP của mô hình có thể nhận được bằng cách lấy trung bình tất cả các giá trị AP. Lấy bài viết này làm ví dụ, chúng tôi có thể tính toán Đối với các giá trị AP của hai mục tiêu đội mũ bảo hiểm và không đội mũ bảo hiểm, chúng tôi tính trung bình các giá trị AP của hai nhóm để thu được giá trị mAP của toàn bộ mô hình. Giá trị càng gần với 1, hiệu suất của mô hình càng tốt.

Để biết thêm các định nghĩa học thuật, bạn có thể xem chúng trên Zhihu hoặc csdn. Lấy mô hình mà chúng tôi đã đào tạo lần này làm ví dụ. Sau khi mô hình kết thúc, bạn sẽ tìm thấy ba hình ảnh đại diện cho tỷ lệ nhớ lại và độ chính xác của mô hình của chúng tôi khi xác nhận đặt. tỷ lệ và mật độ trung bình trung bình.
![image-20211212175851524](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212175851524.png)

Lấy đường cong PR làm ví dụ, bạn có thể thấy rằng mô hình của chúng tôi có mật độ trung bình trung bình là 0,832 trên bộ xác nhận。

![PR_curve](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/PR_curve.png)

Nếu không có đường cong như vậy trong thư mục của bạn, có thể do mô hình của bạn dừng lại giữa chừng trong quá trình đào tạo và không thực hiện quá trình xác minh. Bạn có thể tạo những hình ảnh này bằng lệnh sau。

```bash
python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
```

Cuối cùng, đây là danh sách giải thích chi tiết về các chỉ số đánh giá, đây có thể nói là định nghĩa sơ khai nhất.。

![img](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/20200411141530456.png)

## sử dụng mô hình

Việc sử dụng mô hình đều được tích hợp trong thư mục `detect.py`, bạn có thể tham khảo nội dung mình muốn dò theo hướng dẫn sau

```bash
 # camera phát hiện
 python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source 0  # webcam
 # Phát hiện tệp hình ảnh
  python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source file.jpg  # image 
 # Phát hiện các tệp video
   python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source file.mp4  # video
 # Phát hiện các tệp trong một thư mục
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt path/  # directory
 # Phát hiện video trên web
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'https://youtu.be/NUsoVlDFqZg'  # YouTube video
 # Phát hiện phát trực tuyến
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream                            
```

Ví dụ: lấy mô hình mặt nạ của chúng tôi làm ví dụ, nếu chúng tôi thực thi`python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source  data/images/fishman.jpg`Lệnh có thể nhận được kết quả kiểm tra như vậy。

![fishman](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/fishman.jpg)

## Xây dựng giao diện trực quan

Phần của giao diện trực quan nằm trong tệp `window.py`, là thiết kế giao diện được hoàn thành bởi pyqt5. Trước khi bắt đầu giao diện, bạn cần thay thế mô hình bằng mô hình mà bạn đã đào tạo. Vị trí thay thế nằm trong phần thứ 60 của `window.py` OK, sửa đổi nó thành địa chỉ mô hình của bạn. Nếu bạn có GPU, bạn có thể đặt thiết bị thành 0, có nghĩa là sử dụng GPU dòng 0, có thể tăng tốc độ nhận dạng mô hình.。

![image-20211212194547804](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212194547804.png)

Sau khi thay xong, nhấp chuột phải để chạy để khởi động giao diện đồ họa, bạn hãy tự mình chạy thử để xem hiệu quả.

![image-20211212194914890](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212194914890.png)

## Tìm tôi

Bạn có thể tìm thấy tôi theo những cách này。

Facebook: [Link](https://www.facebook.com/lekien4562)

Youtube：[Link](https://www.youtube.com/channel/UC-dzQg6_ub6DkYBB2HPJB6g)

Theo dõi tôi ngay bây giờ！















