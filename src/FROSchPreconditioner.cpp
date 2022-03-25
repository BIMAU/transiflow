#ifndef FVM_FROSCH_PRECONDITIONER_CPP
#define FVM_FROSCH_PRECONDITIONER_CPP

namespace FROSch {

    IfpackPreconditioner::IfpackPreconditioner(Teuchos::RCP<const Epetra_RowMatrix> matrix,
                                               Teuchos::RCP<Teuchos::ParameterList> &parameterList):
        Comm_ (Teuchos::rcp(&(matrix->Comm()))),
        Matrix_ (matrix),
        ParameterList_ (parameterList)
    {
        using namespace Teuchos;
        using namespace Xpetra;
        using Xpetra::Matrix;

        const Epetra_MpiComm& tmpComm = dynamic_cast<const Epetra_MpiComm&> (*Comm_);
        TeuchosComm_ = rcp(new MpiComm<int> (tmpComm.Comm()));

        // Convert the matrix
        RCP<const Epetra_CrsMatrix> crsMatrix = rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix);
        RCP<Epetra_CrsMatrix> crsMatrix_nonconst = rcp_const_cast<Epetra_CrsMatrix>(crsMatrix);
        RCP<CrsMatrix<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > xCrsMatrix = rcp(new EpetraCrsMatrixT<FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(crsMatrix_nonconst));
        // this is a wrapper to turn the object into an Xpetra object
        RCP<const Matrix<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > xMatrix = rcp(new CrsMatrixWrap<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(xCrsMatrix));
        // this is an Xpetra::Matrix that allows 'viewing' the matrix like a block matrix, for instance

        FROSchPreconditioner_.reset(new TwoLevelBlockPreconditioner<double,int>(xMatrix,ParameterList_));
        //ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
    }

    int IfpackPreconditioner::Initialize()
    {
        using namespace Teuchos;
        using namespace Xpetra;

        unsigned dimension = ParameterList_->get("Dimension",3);

        ArrayRCP<unsigned> dofsPerNodeVector = ParameterList_->get("DofsPerNodeVector",Teuchos::null);
        FROSCH_ASSERT(!dofsPerNodeVector.is_null(),"dofsPerNodeVector.is_null()");

        ArrayRCP<FROSch::DofOrdering> dofOrderings = ParameterList_->get("DofOrderings",Teuchos::null);
        FROSCH_ASSERT(!dofOrderings.is_null(),"dofOrderings.is_null()");

        unsigned overlap = ParameterList_->get("Overlap",1);

        ArrayRCP<RCP<const Epetra_Map> > repeatedMaps = ParameterList_->get("RepeatedMaps",Teuchos::null);
        FROSCH_ASSERT(!repeatedMaps.is_null(),"repeatedMaps.is_null()");
        ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > repeatedMapsX(repeatedMaps.size());
        for (size_t i = 0; i < repeatedMaps.size(); i++) {
            FROSCH_ASSERT(!repeatedMaps[i].is_null(),"repeatedMaps[i].is_null()");
            repeatedMapsX[i] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *repeatedMaps[i], TeuchosComm_ );
        }

        ArrayRCP<ArrayRCP<RCP<const Epetra_Map> > > dofMaps = ParameterList_->get("DofMaps",Teuchos::null);
        FROSCH_ASSERT(!dofMaps.is_null(),"dofMaps.is_null()");
        ArrayRCP<ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > > dofMapsX(dofMaps.size());
        for (size_t i = 0; i < dofMaps.size(); i++) {
            FROSCH_ASSERT(!dofMaps[i].is_null(),"dofMaps[i].is_null()");
            ArrayRCP<RCP<const Map<int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > > tmpMaps(dofMaps[i].size());
            for (size_t j = 0; j < dofMaps[i].size(); j++) {
                FROSCH_ASSERT(!dofMaps[i][j].is_null(),"dofMaps[i][j].is_null()");
                tmpMaps[j] = FROSch::ConvertToXpetra<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>::ConvertMap( Xpetra::UseEpetra, *dofMaps[i][j], TeuchosComm_ );
            }
            dofMapsX[i] = tmpMaps;
        }

        FROSchPreconditioner_->initialize(dimension,
                                          dofsPerNodeVector,
                                          dofOrderings,
                                          overlap,
                                          repeatedMapsX,
                                          null,
                                          null,
                                          dofMapsX);
        IsInitialized_ = true;
        return 0;
    }

    int IfpackPreconditioner::Compute()
    {
        FROSCH_ASSERT(IsInitialized_==true,"IsInitialized_==false.");
        FROSchPreconditioner_->compute();
        IsComputed_ = true;
        return 0;
    }

    int IfpackPreconditioner::ApplyInverse(const Epetra_MultiVector &X,
                                           Epetra_MultiVector &Y) const
    {
        using namespace Teuchos;
        using namespace Xpetra;

        FROSCH_ASSERT(IsComputed_==true,"IsInitialized_==false.");

        RCP<const MultiVector<double,int,FROSch::DefaultGlobalOrdinal> > xX = rcp(new EpetraMultiVectorT<FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(rcpFromRef(const_cast<Epetra_MultiVector&>(X))));
        RCP<MultiVector<double,int,FROSch::DefaultGlobalOrdinal> > xY = rcp(new EpetraMultiVectorT<FROSch::DefaultGlobalOrdinal, KokkosClassic::DefaultNode::DefaultNodeType>(rcpFromRef(Y)));

        FROSchPreconditioner_->apply(*xX,*xY);
        return 0;
    }

} // namespace FROSch

#endif
