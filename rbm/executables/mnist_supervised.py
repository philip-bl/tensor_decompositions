
# tb_dir = mkdtemp()
# print(f"tb_dir = {tb_dir}")
# with tb.SummaryWriter(tb_dir) as tb_writer:
#     visible, hidden = model.gibbs_sample(batch_size=2, num_steps=100000, tb_writer=tb_writer, tb_tag_many_records=f"gibbs_at_mll_-163", tb_tag_one_record=f"gibbs_at_mll_-163")
#     tb_writer.add_image(
#         gibbs_sampling_tag_one_record,
#         train_dataset.extract_images(visible[0]).squeeze(),
#         0,
#         dataformats="HW",
#     )
