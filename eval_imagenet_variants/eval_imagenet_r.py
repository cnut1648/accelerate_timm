from pathlib import Path
import sys
import torch
import torchvision
import torchmetrics
from tqdm.auto import tqdm
from utils import get_args, load_model

pwd = Path(__file__).parent.resolve()
sys.path.insert(0, str(pwd.parent))
from classnames_imagenet import subset100
from classnames_imagenet import classnames_simple as classnames


@torch.no_grad()
def evaluate(model, test_dataloader, n_cls):
    all_wnids = sorted(list(classnames.keys()))
    imagenet_r_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178',
                        'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366',
                        'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546',
                        'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747',
                        'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570',
                        'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238',
                        'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585',
                        'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166',
                        'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341',
                        'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367',
                        'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604',
                        'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486',
                        'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366',
                        'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521',
                        'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855',
                        'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020',
                        'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426',
                        'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734',
                        'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441',
                        'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741',
                        'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883',
                        'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257',
                        'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614',
                        'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704',
                        'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866',
                        'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537',
                        'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940',
                        'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052',
                        'n09472597', 'n09835506', 'n10565667', 'n12267677'}

    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]
    indices_in_1k = [i for i, x in enumerate(imagenet_r_mask) if x]

    if n_cls == 100:
        thousand_wnid = sorted(list(classnames.keys()))
        hundred_wnid = sorted(subset100)
        thousand2hundred = {}
        for i in range(len(thousand_wnid)):
            if thousand_wnid[i] in hundred_wnid:
                thousand2hundred[i] = hundred_wnid.index(thousand_wnid[i])
        indices_in_output = [thousand2hundred[i] for i in indices_in_1k if i in thousand2hundred.keys()]
        twohundred2output = {}
        for i in range(len(indices_in_1k)):
            if indices_in_1k[i] in thousand2hundred.keys():
                index_in_1h = thousand2hundred[indices_in_1k[i]]
                if index_in_1h in indices_in_output:
                    twohundred2output[i] = indices_in_output.index(index_in_1h)
    elif n_cls == 1000:
        indices_in_output = indices_in_1k
    else:
        raise ValueError("n_cls should be either 100 or 1000")


    model.eval().cuda()
    top1 = torchmetrics.Accuracy(task="multiclass", num_classes=len(indices_in_output), top_k=1).to("cuda")
    top5 = torchmetrics.Accuracy(task="multiclass", num_classes=len(indices_in_output), top_k=5).to("cuda")
    for step, (inputs, targets) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        inputs = inputs.cuda()
        outputs = model(inputs)[:, indices_in_output]
        if n_cls == 100:
            targets.apply_(lambda x: twohundred2output.get(x, -1))
            keep_indices = (targets != -1).nonzero(as_tuple=True)[0]
            targets = targets[keep_indices]
            outputs = outputs[keep_indices]
            if len(targets) == 0:
                continue
        preds = torch.argmax(outputs, dim=1)
        targets = targets.cuda()
        top1.update(preds, targets)
        top5.update(outputs, targets)

    print(f"The top1 accuracy is {top1.compute() * 100}%")
    print(f"The top5 accuracy is {top5.compute() * 100}%")


if __name__ == "__main__":
    args = get_args()

    model = load_model(args)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean, std)])

    dataset = torchvision.datasets.ImageFolder(root=str(pwd /"imagenet-r/"), transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluate(model, test_dataloader, args.num_classes)
