# Cultural Branch Evaluation Plan — Bản cho Annotator

Hướng dẫn thực thi đánh giá kênh LCQM của ViG. Bản này dành cho người trực tiếp làm annotation.

---

## ⚠️ ĐỌC TRƯỚC — Vai trò của con người và của AI

**Mọi phán đoán văn hóa và mọi nhãn cuối cùng trong tài liệu này phải do con người tự làm thủ công.**

AI (nếu dùng) chỉ đóng **vai trò bán tự động hỗ trợ**, giới hạn ở hai việc cơ học: (1) mine danh sách cụm từ ứng viên từ corpus theo thống kê, và (2) gom cụm ứng viên gần nghĩa để bạn duyệt nhanh theo nhóm. AI **KHÔNG** quyết định cụm nào là văn hóa, **KHÔNG** gán facet, **KHÔNG** chấm caption đúng/sai, **KHÔNG** thay bạn phân tích lỗi.

Nếu trong quá trình làm bạn có dùng AI để đọc/tóm tắt tài liệu, hãy luôn tự nhắc: AI chỉ sàng lọc thô, còn việc đọc image, đọc caption, đối chiếu, và ra nhãn là **bạn tự làm bằng mắt và bằng hiểu biết văn hóa của bạn**. Lý do: đây là cultural annotation — giá trị khoa học nằm ở chính phán đoán của native annotator. Nếu để AI ra nhãn thay, kết quả không còn là cultural ground truth, và Cohen's kappa giữa hai người mất ý nghĩa (vì cả hai cùng copy một nguồn máy).

Quy tắc đơn giản: **AI sàng — người duyệt và quyết.**

---

## Bối cảnh: ViG và kênh LCQM đo cái gì

ViG thêm module LCQM (Local-Cultural Query Memory) gồm 16 learnable slot, học chuyên biệt hóa về các pattern văn hóa Việt qua contrastive loss với cultural phrase. LCQM tuyên bố ba điều, và công việc của bạn là kiểm chứng từng điều:

1. ViG recover được cultural entity mà baseline bỏ sót (kiểm bằng Task 1, 2).
2. Việc recover KHÔNG kéo theo bịa thêm cultural term (kiểm bằng Task 2, LOR trong Task 5).
3. Mỗi slot chuyên về một nhóm văn hóa (kiểm bằng Task 4 — phần này mình tự extract, không cần bạn annotate).

---

## Task 0 — Xây Vietnamese Cultural Vocabulary

**Mục tiêu.** Tạo `V_cultural`: danh sách cụm từ văn hóa tiếng Việt đã được người audit, có gán facet. Đây là tài nguyên nền, dùng cho cả training lẫn evaluation. Làm được ngay, không cần chờ model.

**Lưu ý nền tảng — segmenter.** Cultural vocabulary phải được tách từ bằng **VnCoreNLP RDRSegmenter** (đúng segmenter mà pipeline GRIT-KTVIC và PhoBERT đang dùng), KHÔNG dùng underthesea. Lý do ở mục "Template & segmenter" cuối tài liệu. Cụm từ lưu dưới dạng gạch dưới như `nón_lá`, `áo_dài`.

**Bốn bước — rõ phần máy làm, phần bạn làm.**

Bước 1 — *(máy làm)* Mine ứng viên. Quét 5 cột reference của 558 dòng test + toàn bộ training caption (đã segment bằng RDRSegmenter). Lọc cụm danh từ theo tần suất ≥ 5 và IDF (bỏ cụm quá phổ biến như người, đường, nhà). Ra ~400-500 cụm thô. Bạn nhận file danh sách này.

Bước 2 — *(máy hỗ trợ, bạn duyệt)* Gom cụm để duyệt nhanh. Máy encode mỗi ứng viên bằng PhoBERT và gom thành nhóm gần nghĩa, để bạn duyệt theo cụm thay vì đọc phẳng 500 dòng. **Nhóm chỉ là gợi ý sắp xếp — bạn vẫn đọc từng cụm.**

Bước 3 — *(BẠN làm thủ công, 2 người)* Audit binary cultural / non-cultural. Hai người Việt có hiểu biết văn hóa đọc **từng** cụm, tự đánh dấu: đây có phải nét văn hóa Việt không.
- Cultural: nón lá, áo dài, áo bà ba, xích lô, cờ đỏ sao vàng, đèn lồng, trống, mặt nạ...
- Non-cultural: căn nhà, kệ hàng, bờ sông...
- Hai người làm **độc lập**, không nhìn bài nhau. Sau đó tính **Cohen's kappa**; cần ≥ 0.7. Chỗ nào bất đồng thì hai người họp lại resolve.

Bước 4 — *(BẠN làm thủ công)* Gán facet cho cụm cultural: trang phục (nón lá, áo dài), ẩm thực (phở, bánh mì), nghi lễ (trống, hương), phương tiện (xích lô, ghe), biểu tượng (cờ đỏ sao vàng), kiến trúc-cảnh quan (mái ngói cong).

**Seed để khởi động.** {nón lá, áo dài, áo bà ba, xích lô, mặt nạ, cờ, trống} — bắt đầu từ đây an toàn hơn mine trắng.

**Output.** `V_cultural` ~200-250 cụm dạng `{cụm: {facet, variants}}`, kèm báo cáo kappa.

**Đạt khi:** kappa ≥ 0.7; vocabulary cover được seed; số cụm hợp lý (~200-250).

---

## Task 1 — Cultural Entity Recall Probe

**Mục tiêu.** Đo ViG recover bao nhiêu cultural entity baseline bỏ sót.

**Cách làm.** Với mỗi concept trong `V_cultural`, tìm tập image có reference nhắc tới. Với mỗi image, kiểm tra từng model có nói concept đó không (matching bằng PhoBERT similarity để bắt cả paraphrase như "chiếc nón" ↔ "nón lá"). Tính recall theo concept, theo model. Phần baseline (A/B/C) chạy được ngay từ CSV.

**Ví dụ thực tế.** nón lá ở image 403, 211: baseline A bỏ sót 403 (nói "tạo dáng chụp ảnh"), B/C recover 403; cả ba bỏ sót 211. → LCR_nón_lá: A=0%, B=50%, C=50%. cờ đỏ sao vàng ở 1339: cả ba bỏ ("kệ hàng"). xích lô ở 331: cả ba recover → đây là positive control, ViG không được tệ hơn.

**Đạt khi:** có bảng recall per concept × model; ViG cao hơn baseline ở concept yếu, giữ nguyên positive control.

---

## Task 2 — Error Taxonomy Delta

**Mục tiêu.** Kiểm ViG có giảm lỗi liên quan văn hóa so baseline không, trên cùng 100 mẫu, cùng bộ taxonomy.

**Cách làm.** Thêm ViG làm Model D, annotate 100 mẫu đó bằng đúng taxonomy đã dùng cho baseline. **Làm blind** (không biết caption nào của model nào khi gán nhãn). Theo dõi ba tag chính: cultural_entity_missed (phải giảm), template_bias (phải giảm), object_hallucination (không được tăng).

**Đây là task BẠN làm thủ công.** AI không gán nhãn lỗi. Bạn đọc image, đọc caption, tự phán đoán lỗi.

**Ví dụ.** Nhóm image baseline collapse thành template giống hệt (vd image 86: cả ba model đều ra "món đồ trang trí trong cửa hàng"). ViG phá được collapse ở những image này là datapoint mạnh.

**Đạt khi:** có bảng đếm lỗi A vs C vs D (ViG). Mong: cultural_entity_missed và template_bias giảm; object_hallucination không tăng mạnh. Nếu hallucination tăng → báo lại, ghi nhận trung thực dù bất lợi.

---

## Task 3 — Paired Comparison

**Mục tiêu.** Đặt cạnh nhau caption baseline vs ViG trên ca văn hóa, gồm cả ca thành công lẫn thất bại.

**Cách làm.** Với mỗi ca: image, reference, caption baseline, caption ViG, và annotation thủ công theo schema (đúng object, action, độ cụ thể, hallucination, bỏ sót). **Bạn tự đối chiếu bằng mắt**, không nhờ AI viết nhận xét thay.

**Ví dụ.** Ca thành công: 403 (nón lá). Ca khó nhiều marker: 563 (áo dài + trống + nghi lễ). Ca thất bại trung thực: 211 hoặc 1096 (nón lá chìm vào generic) — phải đưa cả ca thất bại, không chỉ chọn ca đẹp.

**Đạt khi:** mỗi ca có annotation thủ công đặt cạnh baseline; delta thấy rõ qua side-by-side.

---

## Task 4 — Mechanism Analysis (mình tự làm, không cần bạn annotate)

Ghi ở đây để bạn biết bức tranh đầy đủ. Mình sẽ extract từ trained model: bảng top-5 phrase mỗi slot (xem 16 slot có cluster theo facet không) và phân bố η_loc trên cultural token vs function token (xem có tách ≥ 0.10 không). Không có annotation thủ công ở task này.

---

## Task 5 — LCR/LOR Diagnostic (mình chạy script, không cần bạn annotate)

Mình tính LCR/LOR bằng PhoBERT matching trên `V_cultural` bạn xây. Đây là metric scalar đi kèm, ở mức diagnostic. Lưu ý trung thực sẽ ghi trong paper: hai metric này dùng reference làm proxy cho ground truth thị giác, không phải annotation thị giác độc lập — chuẩn mạnh hơn (annotate cái thực sự thấy trong ảnh) để dành làm future work.

---

## Template cultural vocabulary & segmenter (đọc kỹ trước khi làm Task 0)

**Vì sao phải dùng RDRSegmenter của VnCoreNLP, không phải underthesea.** GRIT khi chạy trên KTVIC tách từ tiếng Việt bằng **VnCoreNLP RDRSegmenter** trước khi đưa vào decoder, nên token trong prediction có dạng gạch dưới (`nón_lá`, `áo_dài`). PhoBERT cũng được pretrain trên text segment bằng đúng RDRSegmenter. Vậy nếu cultural vocab tách bằng underthesea, cụm từ sẽ segment khác kiểu — lệch cả với token decoder sinh ra lẫn với input PhoBERT mong đợi — làm hỏng bước matching. Dùng cùng một segmenter (RDRSegmenter) cho cả ba phía là điều kiện để pipeline khớp.

**Template file.** `V_cultural` lưu dạng JSON, mỗi cụm một entry:

```json
{
  "nón_lá":          {"facet": "trang_phục",  "variants": ["nón", "chiếc_nón_lá"]},
  "áo_dài":          {"facet": "trang_phục",  "variants": ["tà_áo_dài"]},
  "áo_bà_ba":        {"facet": "trang_phục",  "variants": []},
  "xích_lô":         {"facet": "phương_tiện", "variants": ["xe_xích_lô"]},
  "cờ_đỏ_sao_vàng":  {"facet": "biểu_tượng",  "variants": ["lá_cờ", "quốc_kỳ"]},
  "trống":           {"facet": "nghi_lễ",     "variants": ["trống_đồng"]},
  "đèn_lồng":        {"facet": "trang_trí",   "variants": []}
}
```

`facet` là nhóm văn hóa bạn gán ở Task 0 bước 4. `variants` là các cách viết khác bạn thấy trong reference (giúp matching bắt paraphrase). Không có "count" hay "index" — file này dùng để ĐO caption, khác hoàn toàn với vocab.json của GRIT (file đó là {token: count}, dùng để model SINH chữ, không liên quan task của bạn).
