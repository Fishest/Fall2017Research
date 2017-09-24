from base_model import BaseModel

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'


    def initialize(self, opt):
        super().initialize(opt)

        # define the input pairs
        self.input_A = self.Tensor(
            opt.batchSize,
            opt.input_nc,
            opt.fineSize,
            opt.fineSize,
        )
        self.input_B = self.Tensor(
            opt.batchSize,
            opt.output_nc,
            opt.fineSize,
            opt.fineSize,
        )

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.which_model_net_G,
            opt.norm,
            not opt.no_dropout,
            self.gpu_ids,
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ngf,
                opt.which_model_net_G,
                opt.norm,
                not opt.no_dropout,
                self.gpu_ids,
            )

if __name__ == "__main__":
    pix2pix = Pix2PixModel()
    random = None
    pix2pix.initialize(random)
