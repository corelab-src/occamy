configure_file(run.py run.py COPYONLY)
configure_file(dataset.py dataset.py COPYONLY)
configure_file(ops.py ops.py COPYONLY)
configure_file(srgan_model.py srgan_model.py COPYONLY)

file(COPY DIV2K_LR_IMAGE DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY model DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/result)
