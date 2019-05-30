# Eigenface mixture and morphing between faces

_FaceMasters-Application_ loads the warped faces of the team members (i.e. templates matched to scanned faces so that the number of vertices and landmarks match) from a specified folder and performs PCA on this dataset. The sliders in the UI can then be used to explore the results of giving different eigenfaces (sorted in order of decreasing eigenvalues) the specified weights. Furthermore, a second set of sliders can be used to morph between a pair of the loaded faces directly.

**Note**: the shader and skin texture used in the program are taken from [this blog post](http://www.alecjacobson.com/weblog/?p=4827)